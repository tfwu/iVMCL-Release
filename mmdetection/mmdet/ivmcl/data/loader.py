import os
import os.path
import json

from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

from .transform import (get_train_transform, get_val_transform,
                        NCropsTransform)
from .prefetcher import (
    fast_collate, fast_collate_twocrop, fast_collate_n_crop)
from .ordered_distributed_sampler import OrderedDistributedSampler


class ImageFolder_ImgIndex(datasets.ImageFolder):
    # return image index
    def __getitem__(self, index):
        img, target = super(ImageFolder_ImgIndex, self).__getitem__(index)
        return img, target, index


# from torchvision
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# from torchvision
# TODO: specify the return type


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

# from torchvision


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder_MultiLabels(datasets.ImageFolder):
    """ Real labels evaluator for ImageNet
    Paper: `Are we done with ImageNet?` - https://arxiv.org/abs/2006.07159
    Based on Numpy example at https://github.com/google-research/reassessed-imagenet
    """

    def __init__(self,
                 real_json: str,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,):
        super(ImageFolder_MultiLabels, self).__init__(root, transform, target_transform,
                                                      loader, is_valid_file)
        assert os.path.exists(real_json)

        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
            real_labels = {
                f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(real_labels)}

        self.real_labels = real_labels

        assert len(self.imgs) == len(self.real_labels)

        m = 0
        for labels in real_labels.values():
            m = max(m, len(labels))
        self.max_num_labels = m

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = super(ImageFolder_MultiLabels, self).__getitem__(index)

        filename = os.path.basename(self.imgs[index][0])
        targets = self.real_labels[filename]
        if self.target_transform is not None:
            targets = [self.target_transform(t) for t in targets]
        targets = targets + \
            [-1 for _ in range(self.max_num_labels - len(targets))]

        return img, target, targets


def get_train_loader(cfg, train_cfg, distributed,
                     num_crop=0, no_prefetcher=False,
                     fast_collate_mixup=None, with_img_index=False):

    # train_cfg = cfg.data_cfg['train_cfg']

    train_transform = get_train_transform(train_cfg, no_prefetcher)
    if num_crop > 1:
        train_transform = NCropsTransform(train_transform, num=num_crop)

    if with_img_index:
        train_dataset = ImageFolder_ImgIndex(
            os.path.join(cfg.data_root, 'train'), train_transform)
    else:
        train_dataset = datasets.ImageFolder(
            os.path.join(cfg.data_root, 'train'), train_transform)

    sampler = DistributedSampler(train_dataset) if distributed else None

    if no_prefetcher:
        collate_fn = torch.utils.data.dataloader.default_collate
    else:
        if num_crop > 1:
            collate_fn = fast_collate_n_crop
        else:
            collate_fn = fast_collate if fast_collate_mixup is None else \
                fast_collate_mixup

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers,
                              sampler=sampler,
                              shuffle=sampler is None,
                              pin_memory=True,
                              drop_last=train_cfg['drop_last'],
                              collate_fn=collate_fn)

    return train_loader


def get_val_loader(cfg, val_cfg, distributed,
                   no_prefetcher=False, real_json=None, with_img_index=False):

    # val_cfg = cfg.data_cfg['val_cfg']
    val_transform = get_val_transform(val_cfg, no_prefetcher)
    data_dir = os.path.join(cfg.data_root, 'val')
    if not os.path.exists(data_dir):
        data_dir = cfg.data_root
    if real_json is not None:
        val_dataset = ImageFolder_MultiLabels(
            real_json, data_dir, val_transform)
    else:
        if with_img_index:
            val_dataset = ImageFolder_ImgIndex(data_dir, val_transform)
        else:
            val_dataset = datasets.ImageFolder(data_dir, val_transform)

    # sampler = DistributedSampler(val_dataset, shuffle=False) if distributed \
    #     else None

    if distributed:
        # This will add extra duplicate entries to result in equal num
        # of samples per-process, will slightly alter validation results
        # from https://github.com/rwightman/pytorch-image-models/blob/078a51dbac2ec4e401e166a3aec0b3c613e6c06f/timm/data/loader.py
        sampler = OrderedDistributedSampler(val_dataset)
    else:
        sampler = None

    if no_prefetcher:
        collate_fn = torch.utils.data.dataloader.default_collate
    else:
        collate_fn = fast_collate

    if hasattr(cfg, 'test_batch_size'):
        batch_size = cfg.test_batch_size
    else:
        batch_size = cfg.batch_size
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=cfg.num_workers,
                            sampler=sampler,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collate_fn)

    return val_loader
