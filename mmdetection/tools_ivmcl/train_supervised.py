from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time
import warnings

# import jason
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    warnings.warn(
        "Please install NVIDIA apex from '+ \
          'https://www.github.com/nvidia/apex to run this example.")
    has_apex = False
from torch.nn.parallel import DistributedDataParallel as DDP1

import mmcv
from mmcv import Config
from mmcv.runner import init_dist
from mmcv.cnn import (AttnBatchNorm2d, AttnGroupNorm)

from mmdet import __version__
from mmdet.apis import set_random_seed
from mmdet.models import build_backbone
from mmdet.utils import collect_env, get_root_logger

from mmdet.ivmcl import (get_scheduler, separate_norm_params,
                         LabelSmoothingCrossEntropy, SoftTargetCrossEntropy,
                         data_prefetcher, data_prefetcher_with_extra_info,
                         get_train_loader, get_val_loader,
                         mixup_batch, FastCollateMixup, AverageMeter, accuracy,
                         accuracy_multi,
                         reduce_tensor, to_python_float)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--start-epoch', type=int,
                        default=1, help='used for resume')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                        help='use NVIDIA amp for mixed precision training')
    parser.add_argument('--amp-static-loss-scale', type=float, default=128.,
                        help='static loss scale for NVIDIA amp')
    parser.add_argument('--print-freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--save-freq', type=int,
                        default=1, help='save frequency')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='only evaluate')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug by running a few iterations')
    parser.add_argument('--profiling', action='store_true',
                        help='run pytorch profiling')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def load_checkpoint(cfg,
                    model,
                    optimizer,
                    scheduler=None,
                    logger=None):

    if logger is not None:
        logger.info("=> Loading checkpoint '{}'".format(cfg.resume_from))

    checkpoint = torch.load(cfg.resume_from, map_location='cpu')

    global best_acc1, best_acc1_reallabels
    best_acc1 = checkpoint['best_acc1']
    best_acc1_reallabels = checkpoint['best_acc1_reallabels']
    cfg.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if cfg.amp_opt_level != 'O0' and \
        checkpoint['amp_opt_level'] != 'O0':
        amp.load_state_dict(checkpoint['amp'])

    if logger is not None:
        logger.info("=> Loaded successfully '{}' (epoch {})".format(
            cfg.resume_from, checkpoint['epoch']))
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(cfg,
                    epoch,
                    model,
                    optimizer,
                    acc,
                    acc_reallabels,
                    scheduler=None,
                    logger=None,
                    is_best=False,
                    is_best_reallabels=False):
    chkpt_file = os.path.join(cfg.work_dir, 'current.pth')
    state = {
        'amp_opt_level': cfg.amp_opt_level,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc1': acc,
        'best_acc1_reallabels': acc_reallabels
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if cfg.amp_opt_level != 'O0':
        state['amp'] = amp.state_dict()
    torch.save(state, chkpt_file)
    # debug
    # torch.load(chkpt_file, map_location='cpu')
    if is_best:
        state = {
            'state_dict': model.state_dict()
        }
        torch.save(state, os.path.join(cfg.work_dir, 'model_best.pth'))
    if is_best_reallabels:
        state = {
            'state_dict': model.state_dict()
        }
        torch.save(state, os.path.join(cfg.work_dir, 'model_best_reallabels.pth'))
    if logger is not None:
        logger.info('==> Saving to{}'.format(chkpt_file))


def load_pretrained(cfg, model, logger=None):
    ckpt = torch.load(cfg.load_from, map_location='cpu')
    if logger is not None:
        logger.info(
            f"==> Loaded pretrained '{cfg.load_from}'")

    # rename pre-trained keys
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.network.'):
            # remove prefix
            state_dict[k[len("module.network."):]] = state_dict[k]
            # delete renamed k
            del state_dict[k]

    model.load_state_dict(state_dict)


# def AttnNorm2Float(module: nn.Module) -> nn.Module:
#     "If `module` is AttnNorm, don't use half precision."
#     if isinstance(module, (AttnBatchNorm2d, AttnGroupNorm)):
#         module.float()
#     for child in module.children():
#         AttnNorm2Float(child)
#     return module
def AttnNorm2Float(model):
    """If a `module` is AttnNorm, don't use half precision.
    """
    for m in model.modules():
        if isinstance(m, (AttnBatchNorm2d, AttnGroupNorm)):
            m.float()
            for child in m.children():
                child.float()
    return model


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.start_epoch is not None:
        cfg.start_epoch = args.start_epoch
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    cfg.amp_opt_level = args.amp_opt_level
    if not has_apex:
        cfg.amp_opt_level = 'O0'
    cfg.amp_static_loss_scale = args.amp_static_loss_scale
    cfg.eval = args.eval
    if cfg.eval:
        assert os.path.isfile(cfg.load_from)
    cfg.debug = args.debug
    cfg.print_freq = args.print_freq if not cfg.debug else 10
    cfg.save_freq = args.save_freq
    cfg.profiling = args.profiling
    if args.seed is None:
        args.seed = 23

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log cfg
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # model
    model = build_backbone(cfg.model)
    logger.info('Model {} created, param count: {:.3f}M'.format(
                 cfg.model['type'],
                 sum([m.numel() for m in model.parameters()]) / 1e6))

    if cfg.debug and dist.get_rank() == 0:
        print(model)

    if cfg.eval:
        load_pretrained(cfg, model, logger)

    if not distributed and len(cfg.gpu_ids) > 1:
        if cfg.amp_opt_level != 'O0':
            logger.warning(
                'AMP does not work well with nn.DataParallel, disabling.' +
                'Use distributed mode for multi-GPU AMP.')
            cfg.amp_opt_level = 'O0'
        model = nn.DataParallel(model, device_ids=list(cfg.gpu_ids)).cuda()
    else:
        model.cuda()

    # data
    fast_collate_mixup = None
    if hasattr(cfg.data_cfg['train_cfg'], 'mix_up_rate') and \
        cfg.data_cfg['train_cfg']['mix_up_rate'] > 0.:
        fast_collate_mixup = FastCollateMixup(
            cfg.data_cfg['train_cfg']['mix_up_rate'],
            cfg.data_cfg['train_cfg']['label_smoothing_rate'],
            cfg.data_cfg['train_cfg']['num_classes']
        )

    train_loader = get_train_loader(
        cfg, cfg.data_cfg['train_cfg'], distributed,
        fast_collate_mixup=fast_collate_mixup)

    real_labels_file = os.path.join(
        cfg.data_root, 'reassessed-imagenet', 'real.json')
    if os.path.exists(real_labels_file):
        val_loader = get_val_loader(cfg, cfg.data_cfg['val_cfg'],
                                    distributed, real_json=real_labels_file)
        real_labels = True
    else:
        logger.info(f'not found {cfg.data_root} {real_labels_file} ' +
                    'consider to download real labels at ' +
                    'https://github.com/google-research/reassessed-imagenet')
        val_loader = get_val_loader(cfg, cfg.data_cfg['val_cfg'], distributed)
        real_labels = False

    # loss
    if hasattr(cfg.data_cfg['train_cfg'], 'mix_up_rate') and \
        cfg.data_cfg['train_cfg']['mix_up_rate'] > 0.:
        criterion_train = SoftTargetCrossEntropy().cuda()
        criterion_val = torch.nn.CrossEntropyLoss().cuda()
    elif hasattr(cfg.data_cfg['train_cfg'], 'label_smoothing_rate') and \
        cfg.data_cfg['train_cfg']['label_smoothing_rate'] > 0.:
        criterion_train = LabelSmoothingCrossEntropy(
            cfg.data_cfg['train_cfg']['label_smoothing_rate']
        ).cuda()
        criterion_val = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion_train = torch.nn.CrossEntropyLoss().cuda()
        criterion_val = criterion_train

    # optimizer
    lr = cfg.optimizer['lr']
    lr *= cfg.batch_size * dist.get_world_size() / cfg.autoscale_lr_factor
    if hasattr(cfg.optimizer, 'remove_norm_weigth_decay') and \
        cfg.optimizer['remove_norm_weigth_decay']:
        norm_params, base_params = separate_norm_params(model)
        optimizer = torch.optim.SGD([
            {'params': base_params,
                'weight_decay': cfg.optimizer['weight_decay']},
            {'params': norm_params, 'weight_decay': 0.0}],
                                    lr=lr,
                                    momentum=cfg.optimizer['momentum'],
                                    nesterov=cfg.optimizer['nesterov'])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=cfg.optimizer['momentum'],
                                    weight_decay=cfg.optimizer['weight_decay'],
                                    nesterov=cfg.optimizer['nesterov'])

    if cfg.amp_opt_level != 'O0':
        loss_scale = cfg.amp_static_loss_scale if cfg.amp_static_loss_scale  \
            else 'dynamic'
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=cfg.amp_opt_level,
                                          loss_scale=loss_scale,
                                          verbosity=1)
        model = AttnNorm2Float(model)


    if distributed:
        if cfg.amp_opt_level != 'O0':
            model = DDP(model, delay_allreduce=True)
        else:
            model = DDP1(model, device_ids=[args.local_rank])

    if cfg.profiling:
        x = torch.randn((2, 3, 224, 224), requires_grad=True).cuda()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            model(x)
        prof.export_chrome_trace(os.path.join(cfg.work_dir, 'profiling.log'))
        logger.info(f"{prof}")
        return

    # scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), cfg)

    # eval
    if cfg.eval:
        validate(val_loader, model, criterion_val, cfg, logger, distributed,
                 real_labels=real_labels)
        return

    # optionally resume from a checkpoint
    if cfg.resume_from:
        assert os.path.isfile(cfg.resume_from)
        load_checkpoint(cfg, model, optimizer, scheduler, logger)

    # training
    for epoch in range(cfg.start_epoch, cfg.total_epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        train(epoch, train_loader, model,
              criterion_train, optimizer, scheduler, cfg, logger, distributed)
        used_time = time.time() - tic
        remaining_time = (cfg.total_epochs - epoch) * used_time / 3600
        logger.info(
            f'epoch {epoch}, total time {used_time:.2f} sec, estimated remaining time {remaining_time:.2f} hr')

        if real_labels is not None:
            test_acc, is_best, _, is_best_reallabels = validate(
                val_loader, model, criterion_val, cfg, logger, distributed,
                real_labels=real_labels)

            if dist.get_rank() == 0 and (epoch % cfg.save_freq == 0 or is_best or is_best_reallabels):
                save_checkpoint(cfg, epoch, model, optimizer,
                                best_acc1, best_acc1_reallabels,
                                scheduler, logger, is_best, is_best_reallabels)
        else:
            test_acc, is_best = validate(
                val_loader, model, criterion_val, cfg, logger, distributed)
            if dist.get_rank() == 0 and (epoch % cfg.save_freq == 0 or is_best):
                save_checkpoint(cfg, epoch, model, optimizer,
                                best_acc1, best_acc1_reallabels,
                                scheduler, logger, is_best, False)

        if cfg.debug:
            break

    # rename folder
    if dist.get_rank() == 0:
        os.rename(cfg.work_dir, cfg.work_dir+f'-top1-{best_acc1:.2%}')

def train(epoch,
          train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          cfg,
          logger,
          distributed):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    x, y = prefetcher.next()
    idx = 0

    while x is not None:
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(x)

        # loss
        loss = criterion(output, y)

        # backprop
        optimizer.zero_grad()
        if cfg.amp_opt_level != 'O0':
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, y, topk=(1, 5))

        if distributed:
            reduced_loss = reduce_tensor(loss.data)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
        else:
            reduced_loss = loss.data

        top1.update(to_python_float(acc1), x.size(0))
        top5.update(to_python_float(acc5), x.size(0))
        losses.update(to_python_float(reduced_loss), x.size(0))

        # meters
        batch_time.update(time.time() - end)
        end = time.time()

        x, y = prefetcher.next()
        idx = idx + 1

        # print info
        if idx % cfg.print_freq == 0 or x is None:
            logger.info(
                f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Lr {optimizer.param_groups[0]["lr"]:.5f} \t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})')

        if cfg.debug and idx >= 10:
            break


def validate(val_loader,
            model,
            criterion,
            cfg,
            logger,
            distributed,
            real_labels=False):
    global best_acc1, best_acc1_reallabels

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if real_labels:
        top1_reallabels = AverageMeter()
        top5_reallabels = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        if real_labels:
            prefetcher = data_prefetcher_with_extra_info(val_loader)
            x, y, realy = prefetcher.next()
        else:
            prefetcher = data_prefetcher(val_loader)
            x, y = prefetcher.next()
        idx = 0
        while x is not None:
            # compute output
            output = model(x)
            loss = criterion(output, y)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, y, topk=(1, 5))

            if distributed:
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), x.size(0))
            top1.update(to_python_float(acc1), x.size(0))
            top5.update(to_python_float(acc5), x.size(0))

            if real_labels:
                acc1_reallabels, acc5_reallabels, n = accuracy_multi(
                    output.data, realy, topk=(1, 5))
                if distributed:
                    acc1_reallabels = reduce_tensor(acc1_reallabels)
                    acc5_reallabels = reduce_tensor(acc5_reallabels)
                    n = reduce_tensor(n)
                top1_reallabels.update(
                    to_python_float(acc1_reallabels), to_python_float(n))
                top5_reallabels.update(
                    to_python_float(acc5_reallabels), to_python_float(n))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if real_labels:
                x, y, realy = prefetcher.next()
            else:
                x, y = prefetcher.next()
            idx = idx + 1

            if idx % cfg.print_freq == 0 or x is None:
                if real_labels:
                    logger.info(
                        f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                        f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})\t'
                        f'Acc@1_reallabels {top1_reallabels.val:.3%} ({top1_reallabels.avg:.3%})\t'
                        f'Acc@5_reallabels {top5_reallabels.val:.3%} ({top5_reallabels.avg:.3%})')
                else:
                    logger.info(
                        f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                        f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})')

            if cfg.debug and idx >= 10:
                break

        is_best = False
        if top1.avg > best_acc1:
            best_acc1 = top1.avg
            is_best = True

        logger.info(f' * Acc@1 {top1.avg:.3%} Acc@5 {top5.avg:.3%}')
        logger.info(f' ** BestAcc@1 {best_acc1:.3%}')

        if real_labels:
            is_best_reallabels = False
        if top1_reallabels.avg > best_acc1_reallabels:
            best_acc1_reallabels = top1_reallabels.avg
            is_best_reallabels = True

        logger.info(
            f' * Acc@1_reallabels {top1_reallabels.avg:.3%} Acc@5_reallabels {top5_reallabels.avg:.3%}')
        logger.info(f' ** BestAcc@1_reallabels {best_acc1_reallabels:.3%}')

        return top1.avg, is_best, top1_reallabels.avg, is_best_reallabels

    return top1.avg, is_best


if __name__ == '__main__':
    warnings.filterwarnings(
        "ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    warnings.filterwarnings(
        "ignore", "Corrupt EXIF data.  Expecting to read 4 bytes but only got 0.", UserWarning)
    best_acc1 = 0.
    best_acc1_reallabels = 0.
    main()
