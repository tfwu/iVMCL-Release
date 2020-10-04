# iVMCL-Release

It is mainly built on two great python and PyTorch libraries: [MMCV](https://github.com/open-mmlab/mmcv) (commit fe83261, 10/01/2020) and [MMDetection](https://github.com/open-mmlab/mmdetection) (commit 5b18b94, 10/01/2020). We will synchronize and update the two repos with the remote master repos when significant updates arise.

- [Modifications in MMCV](docs/mmcv_ivmcl.md)
- [Modifications in MMDetection](docs/mmdetection_ivmcl.md)

It also uses codes from the great [PyToch Image Models](https://github.com/rwightman/pytorch-image-models) Github repo.

It includes official PyTorch implementations of

- [x] [AOGNets](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_AOGNets_Compositional_Grammatical_Architectures_for_Deep_Learning_CVPR_2019_paper.pdf) (CVPR2019) for image classification in ImageNet-1000 and object detection and semantic segmentation in MS-COCO.  See the previous implementation at [AOGNet-V2](https://github.com/iVMCL/AOGNet-v2).
- [x] [Attentive Normalization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620069.pdf)(ECCV2020) for image classification in ImageNet-1000 and object detection and semantic segmentation in MS-COCO. See the previous implementation at [AttentiveNorm_Detection](https://github.com/iVMCL/AttentiveNorm_Detection) based on MMDetection v1.x.
- [ ] [Interpretable R-CNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Towards_Interpretable_Object_Detection_by_Unfolding_Latent_Structures_ICCV_2019_paper.pdf) (ICCV2019) for object detection in PASCAL VOC07 and MS-COCO. To be released a.s.a.p. Please stay tuned.

## Model Zoo

- [x] [ImageNet Classification](docs/classification.md)
- [ ] [MS-COCO Detection & Segmentation](docs/detection.md) In progress.

## Training from Scratch

- [x] [ImageNet Classification](docs/classification_training.md)
- [x] [Object Detection and Semantic Segmentation](docs/detection_training.md)

## Installation

It follows the installation instructions in [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md), which are summarized as follows.

### Requirements

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+ ([Anacoda](https://www.anaconda.com/) is recommended)
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [NVIDIA Apex](https://github.com/NVIDIA/apex), which has been integrated in PyTorch 1.6+.

### Install iVMCL-Release

a. Create a conda virtual environment and activate it.

```shell
conda create -n ivmcl-release python=3.7 -y
conda activate ivmcl-release
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.


c. Clone the iVMCL-Release repository.

```shell
git clone https://github.com/iVMCL/iVMCL-Release.git
```

d. Compile mmcv

```shell
cd iVMCL-Release/mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ..
```

e. Compile mmdetection

```shell
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build mmdetection on macOS, replace the last command with

```shell
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

f. Compile [NVIDIA apex](https://github.com/NVIDIA/apex#quick-start) (for image classification)

```shell
cd YOUR_PATH_TO/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

g. Update Pillow-SIMD for faster image IO

```shell
cd mmdetection/scripts_ivmcl
chmod +x ./update_pillow.sh
./update_pillow.sh
```

## Citations

Please consider citing the following papers in your publications if they help your research.

```
@inproceedings{li2019aognets,
  title={AOGNets: Compositional Grammatical Architectures for Deep Learning},
  author={Li, Xilai and Song, Xi and Wu, Tianfu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition {(CVPR)}},
  pages={6220--6230},
  year={2019}
}

@inproceedings{li2020attentive,
  title={Attentive Normalization},
  author={Li, Xilai and Sun, Wei and Wu, Tianfu},
  booktitle={Proceedings of the European Conference on Computer Vision {(ECCV)}},
  year={2020}
}

@inproceedings{wu2019towards,
  title={Towards Interpretable Object Detection by Unfolding Latent Structures},
  author={Wu, Tianfu and Song, Xi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision {(ICCV)}},
  pages={6033--6043},
  year={2019}
}
```

## Contact

Please feel free to report issues and any related problems to Tianfu Wu (twu19 at ncsu dot edu).

## License

AOGNets related codes are under [RESEARCH ONLY LICENSE](LICENSE-AOGNET).
