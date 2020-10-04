# from apex

import torch
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous()
    for i, img in enumerate(imgs):
        # nump_array = np.asarray(img, dtype=np.uint8)
        nump_array = np.array(img) # PyTorch 1.5 reports non-writable warnings
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2) # HWC to CHW
        tensor[i] += torch.from_numpy(nump_array)

    if len(batch[0]) == 3:
        # extra_info: 1) sample index, 2) reassessed_labels
        extra_info = torch.tensor([target[2]
                                for target in batch], dtype=torch.int64)

        return tensor, targets, extra_info

    return tensor, targets


class data_prefetcher:

    def __init__(self,
                 loader,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class data_prefetcher_with_extra_info:

    def __init__(self,
                 loader,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_extra = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_extra = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_extra = self.next_extra.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        extra = self.next_extra
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if extra is not None:
            extra.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, extra


def fast_collate_twocrop(batch):
    imgs1 = [img[0][0] for img in batch]
    imgs2 = [img[0][1] for img in batch]
    w = imgs1[0].size[0]
    h = imgs1[0].size[1]
    num_img = len(imgs1)
    tensor1 = torch.zeros( (num_img, 3, h, w), dtype=torch.uint8).contiguous()
    tensor2 = torch.zeros( (num_img, 3, h, w), dtype=torch.uint8).contiguous()

    for i, img1 in enumerate(imgs1):
        # nump_array1 = np.asarray(img1,     dtype=np.uint8)
        # nump_array2 = np.asarray(imgs2[i], dtype=np.uint8)
        nump_array1 = np.array(img1)  # PyTorch 1.5 reports non-writable warnings
        nump_array2 = np.array(imgs2[i])
        if (nump_array1.ndim < 3):
            nump_array1 = np.expand_dims(nump_array1, axis=-1)
            nump_array2 = np.expand_dims(nump_array2, axis=-1)

        nump_array1 = np.rollaxis(nump_array1, 2)
        tensor1[i] += torch.from_numpy(nump_array1)

        nump_array2 = np.rollaxis(nump_array2, 2)
        tensor2[i] += torch.from_numpy(nump_array2)

    if len(batch[0]) == 3:
        img_names = [np.frombuffer(p[2].encode(), dtype=np.uint8)
                     for p in batch]
        len_name = len(img_names[0])  # assume they are the same
        tensor_names = torch.zeros(
            (len(img_names), len_name), dtype=torch.uint8).contiguous()
        for i, p in enumerate(img_names):
            tensor_names[i] += torch.from_numpy(p)

        return tensor1, tensor2, tensor_names

    return tensor1, tensor2


class data_prefetcher_twocrop:

    def __init__(self,
                 loader,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_input1, self.next_input2 = next(self.loader)
        except StopIteration:
            self.next_input1 = None
            self.next_input2 = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input1 = self.next_input1.cuda(non_blocking=True)
            self.next_input2 = self.next_input2.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            self.next_input1 = self.next_input1.float()
            self.next_input1 = self.next_input1.sub_(self.mean).div_(self.std)
            self.next_input2 = self.next_input2.float()
            self.next_input2 = self.next_input2.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input1 = self.next_input1
        input2 = self.next_input2
        if input1 is not None:
            input1.record_stream(torch.cuda.current_stream())
        if input2 is not None:
            input2.record_stream(torch.cuda.current_stream())
        self.preload()
        return input1, input2


class data_prefetcher_twocrop_with_img_name:

    def __init__(self,
                 loader,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_input1, self.next_input2, self.next_name = next(self.loader)
        except StopIteration:
            self.next_input1 = None
            self.next_input2 = None
            self.next_name = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input1 = self.next_input1.cuda(non_blocking=True)
            self.next_input2 = self.next_input2.cuda(non_blocking=True)
            self.next_name = self.next_name.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            self.next_input1 = self.next_input1.float()
            self.next_input1 = self.next_input1.sub_(self.mean).div_(self.std)
            self.next_input2 = self.next_input2.float()
            self.next_input2 = self.next_input2.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input1 = self.next_input1
        input2 = self.next_input2
        name = self.next_name
        if input1 is not None:
            input1.record_stream(torch.cuda.current_stream())
        if input2 is not None:
            input2.record_stream(torch.cuda.current_stream())
        if name is not None:
            name.record_stream(torch.cuda.current_stream())
        self.preload()
        return input1, input2, name


def fast_collate_n_crop(batch):
    num_crop = len(batch[0][0])
    imgs = []
    for c in range(num_crop):
        imgs.append([img[0][c] for img in batch])
    w = imgs[0][0].size[0]
    h = imgs[0][0].size[1]
    num_img = len(imgs[0])
    tensors = []
    for c in range(num_crop):
        tensors.append(
            torch.zeros((num_img, 3, h, w), dtype=torch.uint8).contiguous())
    for c in range(num_crop):
        for i, img in enumerate(imgs[c]):
            # nump_array = np.asarray(img,     dtype=np.uint8)
            # PyTorch 1.5 reports non-writable warnings
            nump_array = np.array(img)
            if (nump_array.ndim < 3):
                nump_array = np.expand_dims(nump_array, axis=-1)

            nump_array = np.rollaxis(nump_array, 2)
            tensors[c][i] += torch.from_numpy(nump_array)

    if len(batch[0]) == 3:
        img_names = [np.frombuffer(p[2].encode(), dtype=np.uint8)
                     for p in batch]
        len_name = len(img_names[0])  # assume they are the same
        tensor_names = torch.zeros(
            (len(img_names), len_name), dtype=torch.uint8).contiguous()
        for i, p in enumerate(img_names):
            tensor_names[i] += torch.from_numpy(p)

        tensors.append(tensor_names)

    return tuple(tensors)


class data_prefetcher_n_crop:

    def __init__(self,
                 loader,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)

        self.preload()

    def preload(self):
        try:
            self.next_inputs = next(self.loader)
        except StopIteration:
            self.next_inputs = [None]
            return
        with torch.cuda.stream(self.stream):
            for i, next_input in enumerate(self.next_inputs):
                next_input = next_input.cuda(non_blocking=True)
                next_input = next_input.float()
                self.next_inputs[i] = next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = []
        for next_input in self.next_inputs:
            if next_input is not None:
                next_input.record_stream(torch.cuda.current_stream())
            inputs.append(next_input)
        self.preload()
        return tuple(inputs)


