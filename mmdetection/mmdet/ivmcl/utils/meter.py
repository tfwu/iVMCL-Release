
import torch
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
        the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def accuracy_multi(output, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for
        the specified values of k

        targets (n by nb_labels): multiple labels (-1 for the dummy label), to handle
            Real labels evaluator for ImageNet
            Paper: `Are we done with ImageNet?` - https://arxiv.org/abs/2006.07159
            Based on Numpy example at https://github.com/google-research/reassessed-imagenet
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        tmp = (targets == -1).sum(1, keepdim=True) == targets.size(1)
        num_effective_samples = batch_size - to_python_float(tmp.sum(0))

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = torch.zeros_like(pred)
        for i in range(targets.size(1)):
            target = targets[:, i: i + 1]
            correct += pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].sum(0, keepdim=True) > 0
            correct_k = correct_k.view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / num_effective_samples))

        num = res[0].new_zeros((1,1))
        num[0] = num_effective_samples

        res.append(num)

        return res


@torch.no_grad()
def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.ones_like(x)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
