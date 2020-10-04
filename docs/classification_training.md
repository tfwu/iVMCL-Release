# Training a Classifier using ImageNet-1K

## [Installation of iVMCL-Release](../README.md#Installation) is completed successfully.

The code has been tested under Ubuntu 16.04 LTS and 18.04 LTS. It also should work in other OS for which MMCV and MMDetection support.

## Dataset Preparation

### ImageNet-1k

- Download the [ImageNet dataset](http://image-net.org/download) to `YOUR_IMAGENET_PATH` and unzip.
    - Move validation images to labeled subfolders
        - This [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.
    - Note: `ILSVRC2015` is used in our training. For the ImageNet-1K calssification task, it is the same as `ILSVRC2012`.
- Create a data subfolder under the cloned `iVMCL-Release/mmdetection` and a symbolic link to the ImageNet dataset
    ```shell
    cd iVMCL-Release/mmdetection
    mkdir data
    cd data
    ln -s YOUR_IMAGENET_PATH ./
    ```
    `E.g.`, the data root (relative) directory will be: `data_root=data/ILSVRC2015/Data/CLS-LOC`.

    The directory structure will look like

    ```
    iVMCL-Release
    ├── mmcv
    ├── mmdetection
        ├── mmdet
        ├── tools
        ├── configs
        ├── tools_ivmcl
        ├── configs_ivmcl
        ├── scripts_ivmcl
        ├── data
        │   ├── ILSVRC2015
        │   │   ├── Annotations
        │   │   ├── ImageSets
        │   │   ├── Data
        │   │       ├── CLS-LOC
        │   │           ├── train
        │   │           ├── test
        |   |           ├── val
        |   |           ├── val_orig
    ```

### ImageNet-1k Reassessed

- Paper: [`Are we done with ImageNet?](https://arxiv.org/abs/2006.07159)
- Download the `real.json` file at the [Reassessed ImageNet](https://github.com/google-research/reassessed-imagenet) repo.
    ```shell
    cd iVMCL-Release/mmdetection/data/ILSVRC2015/Data/CLS-LOC/reassessed-imagenet
    wget https://raw.githubusercontent.com/google-research/reassessed-imagenet/master/real.json
    ```

### Some ImageNet-X validation dataset

- Please run the `imagenet-v2.py`, `imagenet-sketch.py` and `imagenet-adv.py` in the `iVMCL-Release/mmdetection/tools_ivmcl` folder.
    ```shell
    cd iVMCL-Release/mmdetection/tools_ivmcl
    python imagenet-v2.py
    python imagenet-sketch.py
    python imagenet-adv.py
    ```

## Training a model from scratch

- Select a configuration file at the cloned `iVMCL-Release/mmdetection/configs_ivmcl`, or create a new one accordingly.
    - `E.g.`, consider `aognet_12m_an_imagenet.py`
- Check `data_root` in a configuration file to make sure it points to the correct directory
- Change the training hyperparameters if needed, `e.g.`, batch_size
- Run the script to train
    ```shell
    cd iVMCL-Release/mmdetection
    chmod +x ./scripts_ivmcl/*.sh
    ```

    change the GPU configuration in the `train_supervised_dist.sh` accordingly based on your hardware environment.

    ```shell
    ./scripts_ivmcl/train_supervised_dist.sh configs_ivmcl/aognet_12m_an_imagenet.py
    ```
