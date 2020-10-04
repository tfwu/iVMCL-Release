from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import re

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

from pytablewriter import MarkdownTableWriter

import mmcv
from mmcv import Config
from mmcv.runner import init_dist, load_checkpoint
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
                         accuracy_multi, reduce_tensor, to_python_float)


imagenet_x = ['imagenet-1k',
              'imagenetv2-matched-frequency', 'imagenetv2-threshold0.7',
              'imagenetv2-topimages', 'imagenet-sketch',
              'imagenet-a', 'imagenet-o']

# from https://github.com/hendrycks/natural-adv-examples/blob/master/eval.py
adv_thousand_k_to_200 = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: 2, 14: -1, 15: 3, 16: -1, 17: 4, 18: -1, 19: -1, 20: -1, 21: -1, 22: 5, 23: 6, 24: -1, 25: -1, 26: -1, 27: 7, 28: -1, 29: -1, 30: 8, 31: -1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: 9, 38: -1, 39: 10, 40: -1, 41: -1, 42: 11, 43: -1, 44: -1, 45: -1, 46: -1, 47: 12, 48: -1, 49: -1, 50: 13, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: 14, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: 15, 71: 16, 72: -1, 73: -1, 74: -1, 75: -1, 76: 17, 77: -1, 78: -1, 79: 18, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 19, 90: 20, 91: -1, 92: -1, 93: -1, 94: 21, 95: -1, 96: 22, 97: 23, 98: -1, 99: 24, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 25, 106: -1, 107: 26, 108: 27, 109: -1, 110: 28, 111: -1, 112: -1, 113: 29, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: -1, 124: 30, 125: 31, 126: -1, 127: -1, 128: -1, 129: -1, 130: 32, 131: -1, 132: 33, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: 34, 144: 35, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1, 150: 36, 151: 37, 152: -1, 153: -1, 154: -1, 155: -1, 156: -1, 157: -1, 158: -1, 159: -1, 160: -1, 161: -1, 162: -1, 163: -1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: -1, 172: -1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: -1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: -1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: -1, 196: -1, 197: -1, 198: -1, 199: -1, 200: -1, 201: -1, 202: -1, 203: -1, 204: -1, 205: -1, 206: -1, 207: 38, 208: -1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: -1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: -1, 232: -1, 233: -1, 234: 39, 235: 40, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: -1, 243: -1, 244: -1, 245: -1, 246: -1, 247: -1, 248: -1, 249: -1, 250: -1, 251: -1, 252: -1, 253: -1, 254: 41, 255: -1, 256: -1, 257: -1, 258: -1, 259: -1, 260: -1, 261: -1, 262: -1, 263: -1, 264: -1, 265: -1, 266: -1, 267: -1, 268: -1, 269: -1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: -1, 277: 42, 278: -1, 279: -1, 280: -1, 281: -1, 282: -1, 283: 43, 284: -1, 285: -1, 286: -1, 287: 44, 288: -1, 289: -1, 290: -1, 291: 45, 292: -1, 293: -1, 294: -1, 295: 46, 296: -1, 297: -1, 298: 47, 299: -1, 300: -1, 301: 48, 302: -1, 303: -1, 304: -1, 305: -1, 306: 49, 307: 50, 308: 51, 309: 52, 310: 53, 311: 54, 312: -1, 313: 55, 314: 56, 315: 57, 316: -1, 317: 58, 318: -1, 319: 59, 320: -1, 321: -1, 322: -1, 323: 60, 324: 61, 325: -1, 326: 62, 327: 63, 328: -1, 329: -1, 330: 64, 331: -1, 332: -1, 333: -1, 334: 65, 335: 66, 336: 67, 337: -1, 338: -1, 339: -1, 340: -1, 341: -1, 342: -1, 343: -1, 344: -1, 345: -1, 346: -1, 347: 68, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: -1, 354: -1, 355: -1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 69, 362: -1, 363: 70, 364: -1, 365: -1, 366: -1, 367: -1, 368: -1, 369: -1, 370: -1, 371: -1, 372: 71, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: 72, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: 73, 387: -1, 388: -1, 389: -1, 390: -1, 391: -1, 392: -1, 393: -1, 394: -1, 395: -1, 396: -1, 397: 74, 398: -1, 399: -1, 400: 75, 401: 76, 402: 77, 403: -1, 404: 78, 405: -1, 406: -1, 407: 79, 408: -1, 409: -1, 410: -1, 411: 80, 412: -1, 413: -1, 414: -1, 415: -1, 416: 81, 417: 82, 418: -1, 419: -1, 420: 83, 421: -1, 422: -1, 423: -1, 424: -1, 425: 84, 426: -1, 427: -1, 428: 85, 429: -1, 430: 86, 431: -1, 432: -1, 433: -1, 434: -1, 435: -1, 436: -1, 437: 87, 438: 88, 439: -1, 440: -1, 441: -1, 442: -1, 443: -1, 444: -1, 445: 89, 446: -1, 447: -1, 448: -1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: 90, 457: 91, 458: -1, 459: -1, 460: -1, 461: 92, 462: 93, 463: -1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: -1, 470: 94, 471: -1, 472: 95, 473: -1, 474: -1, 475: -1, 476: -1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 96, 484: -1, 485: -1, 486: 97, 487: -1, 488: 98, 489: -1, 490: -1, 491: -1, 492: 99, 493: -1, 494: -1, 495: -1, 496: 100, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1, 511: -1, 512: -1, 513: -1, 514: 101, 515: -1, 516: 102, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: 103, 529: -1, 530: 104, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: 105, 540: -1, 541: -1, 542: 106, 543: 107, 544: -1, 545: -1, 546: -1, 547: -1, 548: -1, 549: 108, 550: -1, 551: -1, 552: 109, 553: -1, 554: -1, 555: -1, 556: -1, 557: 110, 558: -1, 559: -1, 560: -1, 561: 111, 562: 112, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: 113, 570: -1, 571: -1, 572: 114, 573: 115, 574: -1, 575: 116, 576: -1, 577: -1, 578: -1, 579: 117, 580: -1, 581: -1, 582: -1, 583: -1, 584: -1, 585: -1, 586: -1, 587: -1, 588: -1, 589: 118, 590: -1, 591: -1, 592: -1, 593: -1, 594: -1, 595: -1, 596: -1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: 119, 607: 120, 608: -1, 609: 121, 610: -1, 611: -1, 612: -1, 613: -1, 614: 122, 615: -1, 616: -1, 617: -1, 618: -1, 619: -1, 620: -1, 621: -1, 622: -1, 623: -1, 624: -1, 625: -1, 626: 123, 627: 124, 628: -1, 629: -1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: -1, 638: -1, 639: -1, 640: 125, 641: 126, 642: 127, 643: 128, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: -1, 658: 129, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: 130, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: 131, 678: -1, 679: -1, 680: -1, 681: -1, 682: 132, 683: -1, 684: 133, 685: -1, 686: -1, 687: 134, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 135, 702: -1, 703: -1, 704: 136, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: -1, 718: -1, 719: 137, 720: -1, 721: -1, 722: -1, 723: -1, 724: -1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: 138, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: 139, 747: -1, 748: -1, 749: 140, 750: -1, 751: -1, 752: 141, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: 142, 759: -1, 760: -1, 761: -1, 762: -1, 763: 143, 764: -1, 765: 144, 766: -1, 767: -1, 768: 145, 769: -1, 770: -1, 771: -1, 772: -1, 773: 146, 774: 147, 775: -1, 776: 148, 777: -1, 778: -1, 779: 149, 780: 150, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: 151, 787: -1, 788: -1, 789: -1, 790: -1, 791: -1, 792: 152, 793: -1, 794: -1, 795: -1, 796: -1, 797: 153, 798: -1, 799: -1, 800: -1, 801: -1, 802: 154, 803: 155, 804: 156, 805: -1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: -1, 813: 157, 814: -1, 815: 158, 816: -1, 817: -1, 818: -1, 819: -1, 820: 159, 821: -1, 822: -1, 823: 160, 824: -1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: 161, 832: -1, 833: 162, 834: -1, 835: 163, 836: -1, 837: -1, 838: -1, 839: 164, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: 165, 846: -1, 847: 166, 848: -1, 849: -1, 850: 167, 851: -1, 852: -1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: 168, 860: -1, 861: -1, 862: 169, 863: -1, 864: -1, 865: -1, 866: -1, 867: -1, 868: -1, 869: -1, 870: 170, 871: -1, 872: -1, 873: -1, 874: -1, 875: -1, 876: -1, 877: -1, 878: -1, 879: 171, 880: 172, 881: -1, 882: -1, 883: -1, 884: -1, 885: -1, 886: -1, 887: -1, 888: 173, 889: -1, 890: 174, 891: -1, 892: -1, 893: -1, 894: -1, 895: -1, 896: -1, 897: 175, 898: -1, 899: -1, 900: 176, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 177, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: 178, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: 179, 925: -1, 926: -1, 927: -1, 928: -1, 929: -1, 930: -1, 931: -1, 932: 180, 933: 181, 934: 182, 935: -1, 936: -1, 937: 183, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 184, 944: -1, 945: 185, 946: -1, 947: 186, 948: -1, 949: -1, 950: -1, 951: 187, 952: -1, 953: -1, 954: 188, 955: -1, 956: 189, 957: 190, 958: -1, 959: 191, 960: -1, 961: -1, 962: -1, 963: -1, 964: -1, 965: -1, 966: -1, 967: -1, 968: -1, 969: -1, 970: -1, 971: 192, 972: 193, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 194, 981: 195, 982: -1, 983: -1, 984: 196, 985: -1, 986: 197, 987: 198, 988: 199, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}

adv_indices_in_1k = [k for k in adv_thousand_k_to_200 if adv_thousand_k_to_200[k] != -1]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate classifiers on ImageNet-X')
    parser.add_argument('pretrained_model_dir', help='the dir of pretrained models')
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

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def AttnNorm2Float(model):
    """If a `module` is AttnNorm, don't use half precision.
    """
    for m in model.modules():
        if isinstance(m, (AttnBatchNorm2d, AttnGroupNorm)):
            m.float()
            for child in m.children():
                child.float()
    return model


def test(args, model_dir, distributed):
    global has_imagenet_reassessed

    args.work_dir = model_dir
    model_name = osp.basename(model_dir)
    orig_top1_acc = 0.
    orig_top1_acc_reallabels = 0.
    # remove top1 tag
    tag = '-top1-'
    idx = model_name.find(tag)
    tag1 = '-top1_reallabels-'
    idx1 = model_name.find(tag1)
    if idx >= 0:
        orig_top1_acc = float(model_name[idx + len(tag) : idx1 - 1]) / 100.
        if idx1 > 0:
            orig_top1_acc_reallabels = float(model_name[idx1 + len(tag1) :-2]) / 100.
        model_name = model_name[:idx]

    # len_model_name = max(len_model_name, len(model_name))
    args.config = os.path.join(model_dir, f'{model_name}.py')
    if not os.path.exists(args.config):
        print(f'Not found {args.config}')
        return None

    cfg = Config.fromfile(args.config)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.amp_opt_level = args.amp_opt_level
    if not has_apex:
        cfg.amp_opt_level = 'O0'
    cfg.amp_static_loss_scale = args.amp_static_loss_scale
    cfg.print_freq = args.print_freq

    cfg.seed = args.seed

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'evalImageNetX.log')
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

    # set random seeds
    if args.seed is None:
        args.seed = 23
    logger.info('Set random seed to {}, deterministic: {}'.format(
        args.seed, args.deterministic))
    set_random_seed(args.seed, deterministic=args.deterministic)

    # model
    model = build_backbone(cfg.model)
    num_params = sum([m.numel() for m in model.parameters()]) / 1e6
    logger.info('Model {} created, param count: {:.3f}M'.format(
        model_name, num_params))

    ckpt_file = os.path.join(model_dir, 'current.pth')

    load_checkpoint(model, ckpt_file, logger=logger)

    # ckpt = torch.load(ckpt_file, map_location='cpu')
    # state_dict = ckpt['model']
    # for k in list(state_dict.keys()):
    #     if k.startswith('module.'):
    #         # remove prefix
    #         state_dict[k[len("module."):]] = state_dict[k]
    #         # delete renamed k
    #         del state_dict[k]
    # model.load_state_dict(state_dict)

    if not distributed and len(cfg.gpu_ids) > 1:
        if cfg.amp_opt_level != 'O0':
            logger.warning(
                'AMP does not work well with nn.DataParallel, disabling.' +
                'Use distributed mode for multi-GPU AMP.')
            cfg.amp_opt_level = 'O0'
        model = nn.DataParallel(model, device_ids=list(cfg.gpu_ids)).cuda()
    else:
        model.cuda()

    # loss
    criterion_val = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    lr = cfg.optimizer['lr']
    lr *= cfg.batch_size * dist.get_world_size() / cfg.autoscale_lr_factor
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

    result = [model_name, num_params]
    for dataset in imagenet_x:
        # data
        if dataset == 'imagenet-1k':
            cfg.data_root = 'data/ILSVRC2015/Data/CLS-LOC'
        else:
            cfg.data_root = f'data/{dataset}'
        if not os.path.exists(cfg.data_root):
            logger.info(f'not found {cfg.data_root}')
            continue

        indices_in_1k = None
        if dataset in ['imagenet-a', 'imagenet-o']:
            indices_in_1k = adv_indices_in_1k

        real_labels = False
        if dataset == 'imagenet-1k':
            real_labels_file = os.path.join(
                cfg.data_root, 'reassessed-imagenet', 'real.json')
            if os.path.exists(real_labels_file):
                val_loader = get_val_loader(cfg, cfg.data_cfg['val_cfg'],
                                            distributed, real_json=real_labels_file)
                real_labels = True
                has_imagenet_reassessed = True
            else:
                logger.info(f'not found {cfg.data_root} {real_labels_file} ' +
                            'consider to download real labels at ' +
                            'https://github.com/google-research/reassessed-imagenet')
                val_loader = get_val_loader(cfg, cfg.data_cfg['val_cfg'], distributed)
        else:
            val_loader = get_val_loader(
                cfg, cfg.data_cfg['val_cfg'], distributed)

        # eval
        results = validate(val_loader, model, criterion_val,
                                cfg, logger, distributed,
                                indices_in_1k=indices_in_1k,
                                real_labels=real_labels)

        result.append((round(results[0], 3), round(results[1], 3)))
        logger.info(
            f'** {model_name} - {dataset} top1-acc {results[0]:.3%}, top5-acc {results[1]:.3%}')

        if len(results) == 4:
            result.append((round(results[2], 3), round(results[3], 3)))
            logger.info(
                f'** {model_name} - {dataset} top1-acc_reallabels {results[2]:.3%}, top5-acc_reallabels {results[3]:.3%}')

    return result


def main():
    global has_imagenet_reassessed

    args = parse_args()

    torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        dist_params = dict(backend='nccl', init_method='env://')
        init_dist(args.launcher, **dist_params)

    results = []

    # pretrained_model_dir: a single model
    result = test(args, args.pretrained_model_dir, distributed)
    if result is not None:
        results.append(result)

    # pretrained_model_dir: a parent folder
    for model_dir in os.scandir(args.pretrained_model_dir):
        if not model_dir.is_dir():
            continue
        result = test(args, model_dir, distributed)
        if result is None:
            continue

        results.append(result)

    if dist.get_rank() == 0:
        # save
        mmcv.dump(results, os.path.join(
            args.pretrained_model_dir, 'ImageNet-X-Eval.pkl'))

        # output markdown table
        writer = MarkdownTableWriter()
        writer.table_name = "Top-1 and Top-5 Error Rates"
        writer.headers = ["Model", "Params (M)"]

        for dataset in imagenet_x:
            writer.headers.append(dataset)
            if dataset == 'imagenet-1k' and has_imagenet_reassessed:
                writer.headers.append('imagenet-1k-reassessed')

        writer.value_matrix = results
        writer.margin = 1  # add a whitespace for both sides of each cell
        writer.dump(os.path.join(args.pretrained_model_dir, 'ImageNet-X-Eval-Table.txt'))


def validate(val_loader,
             model,
             criterion,
             cfg,
             logger,
             distributed,
             indices_in_1k=None,
             real_labels=False):

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
            if indices_in_1k is not None:
                output = output[:, indices_in_1k]

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

        logger.info(f' * Acc@1 {top1.avg:.3%} Acc@5 {top5.avg:.3%}')
        if real_labels:
            logger.info(
                f' * Acc@1_reallabels {top1_reallabels.avg:.3%} Acc@5_reallabels {top5_reallabels.avg:.3%}')
            return top1.avg, top5.avg, top1_reallabels.avg, top5_reallabels.avg

    return top1.avg, top5.avg


if __name__ == '__main__':
    warnings.filterwarnings(
        "ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    warnings.filterwarnings(
        "ignore", "Corrupt EXIF data.  Expecting to read 4 bytes but only got 0.", UserWarning)

    has_imagenet_reassessed = False

    main()
