import random

from PIL import ImageFilter
from PIL import Image

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# from pytorch-image-models
def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


class NCropsTransform:
    """Take N random crops of one image as the query and key."""

    def __init__(self, base_transform, num=2):
        self.base_transform = base_transform
        self.num = num

    def __call__(self, x):
        out = []
        for i in range(self.num):
            out.append(self.base_transform(x))
        return out


# from fb moco
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_train_transform(cfg, no_prefetcher=False):

    image_size = cfg['crop_size']
    crop_min_scale = 0.08 if not hasattr(cfg, 'crop_min_scale') \
        else cfg.crop_min_scale

    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if cfg['type'] == 'NULL':
       trans = [
            transforms.RandomResizedCrop(
                image_size, scale=(crop_min_scale, 1.),
                interpolation=_pil_interp(cfg['interpolation'])),
            transforms.RandomHorizontalFlip()
        ]
    elif cfg['type'] == 'CJ':
        trans = [
            transforms.RandomResizedCrop(
                image_size, scale=(crop_min_scale, 1.),
                interpolation=_pil_interp(cfg['interpolation'])),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip()
        ]
    elif cfg['type'] == 'CJ+GBlur':
        trans = [
            transforms.RandomResizedCrop(
                image_size, scale=(crop_min_scale, 1.),
                interpolation=_pil_interp(cfg['interpolation'])),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip()
        ]
    else:
        raise NotImplementedError('augmentation not supported: {}'.format(
            cfg['type']))


    if no_prefetcher:
        trans.append( transforms.ToTensor() )
        trans.append( normalize )

    return transforms.Compose(trans)


def get_val_transform(cfg, no_prefetcher=False):
    image_size = cfg['crop_size']
    crop_padding = cfg['crop_padding']

    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    trans = [
        transforms.Resize(image_size + crop_padding),
        transforms.CenterCrop(image_size)
    ]

    if no_prefetcher:
        trans.append(transforms.ToTensor())
        trans.append(normalize)

    return transforms.Compose(trans)
