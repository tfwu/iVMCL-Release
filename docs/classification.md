# ImageNet-1k Classification

Remarks: All the models are directly converted from the previous implementation, the [AOGNet-V2](https://github.com/iVMCL/AOGNet-v2) repo. They are trained from scratch  There are minor performance difference.


- [x] All the models are released on Google Drive ([link](https://drive.google.com/drive/folders/1puKc5g03bnt1qtzaHLCxu5-8tWmlo_WP?usp=sharing))
- [x] Add the evaluation method proposed in [`Are we done with ImageNet?](https://arxiv.org/abs/2006.07159) See the *imagenet-1k-reassessed* column in the below table.
    - Please refer to [PyTorch Image Model (ImageNet-Reassessed)](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-real.csv) for a comprehensive evaluation of state-of-the-art DNNs.
- [x] Add a few other third-party validation datasets, the [ImageNet V2 dataset](https://github.com/modestyachts/ImageNetV2), the [ImageNet Sketch dataset](https://github.com/HaohanWang/ImageNet-Sketch), the [ImageNet Adversarial dataset](https://github.com/hendrycks/natural-adv-examples).
    - Please refer to [imagenetv2-matched-frequency](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenetv2-matched-frequency.csv), [imagenet-sketch](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-sketch.csv), and [imagenet-a](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-a.csv) in the [PyTorch Image Model](https://github.com/rwightman/pytorch-image-models) repo for comprehensive evaluations of state-of-the-art DNNs.



# Top-1 and Top-5 Accuracy (w/ a normal training setup)
|                  Model                  | Params (M) |  imagenet-1k   | imagenet-1k-reassessed | imagenetv2-matched-frequency | imagenetv2-threshold0.7 | imagenetv2-topimages | imagenet-sketch |   imagenet-a   |   imagenet-o   |
|-----------------------------------------|-----------:|----------------|------------------------|------------------------------|-------------------------|----------------------|-----------------|----------------|----------------|
| aognet_12m_bn_imagenet                  |     12.259 | (0.775, 0.939) | (0.84, 0.961)          | (0.648, 0.861)               | (0.738, 0.929)          | (0.79, 0.953)        | (0.257, 0.447)  | (0.054, 0.318) | (0.017, 0.054) |
| aognet_12m_an_imagenet                  |     12.373 | (0.787, 0.942) | (0.849, 0.964)         | (0.666, 0.873)               | (0.757, 0.936)          | (0.802, 0.959)       | (0.269, 0.454)  | (0.095, 0.385) | (0.019, 0.054) |
| aognet_40m_bn_imagenet                  |     40.153 | (0.802, 0.951) | (0.859, 0.969)         | (0.682, 0.882)               | (0.769, 0.942)          | (0.812, 0.964)       | (0.277, 0.459)  | (0.118, 0.42)  | (0.016, 0.057) |
| aognet_40m_an_imagenet                  |     40.389 | (0.807, 0.953) | (0.861, 0.97)          | (0.692, 0.886)               | (0.779, 0.944)          | (0.823, 0.963)       | (0.287, 0.474)  | (0.157, 0.462) | (0.017, 0.055) |
| resnet_34_bn_imagenet                   |     21.798 | (0.745, 0.919) | (0.812, 0.948)         | (0.62, 0.837)                | (0.708, 0.91)           | (0.765, 0.943)       | (0.233, 0.407)  | (0.022, 0.215) | (0.018, 0.052) |
| resnet_34_an_bn1_imagenet               |     21.996 | (0.755, 0.926) | (0.822, 0.952)         | (0.64, 0.851)                | (0.724, 0.917)          | (0.781, 0.947)       | (0.24, 0.415)   | (0.038, 0.259) | (0.018, 0.058) |
| resnet_34_an_bn12_imagenet              |     22.194 | (0.751, 0.924) | (0.814, 0.95)          | (0.629, 0.845)               | (0.713, 0.912)          | (0.771, 0.944)       | (0.227, 0.398)  | (0.044, 0.271) | (0.016, 0.052) |
| resnet_50_bn_imagenet                   |     25.557 | (0.769, 0.934) | (0.835, 0.958)         | (0.65, 0.857)                | (0.742, 0.928)          | (0.79, 0.952)        | (0.247, 0.422)  | (0.033, 0.267) | (0.02, 0.06)   |
| resnet_50_an_bn2_imagenet               |     25.755 | (0.784, 0.941) | (0.844, 0.961)         | (0.666, 0.865)               | (0.748, 0.929)          | (0.798, 0.955)       | (0.247, 0.422)  | (0.082, 0.349) | (0.017, 0.053) |
| resnet_50_an_bn3_imagenet               |     26.349 | (0.782, 0.94)  | (0.839, 0.961)         | (0.659, 0.863)               | (0.746, 0.93)           | (0.796, 0.955)       | (0.246, 0.416)  | (0.086, 0.36)  | (0.014, 0.049) |
| resnet_50_an_all_imagenet               |     26.947 | (0.778, 0.938) | (0.837, 0.96)          | (0.657, 0.859)               | (0.743, 0.925)          | (0.794, 0.952)       | (0.236, 0.407)  | (0.084, 0.356) | (0.016, 0.048) |
| resnet_101_bn_imagenet                  |     44.549 | (0.788, 0.942) | (0.849, 0.963)         | (0.667, 0.866)               | (0.757, 0.935)          | (0.805, 0.958)       | (0.276, 0.456)  | (0.064, 0.323) | (0.021, 0.059) |
| resnet_101_an_imagenet                  |     45.001 | (0.794, 0.946) | (0.852, 0.965)         | (0.675, 0.877)               | (0.759, 0.936)          | (0.806, 0.959)       | (0.277, 0.456)  | (0.118, 0.404) | (0.017, 0.051) |
| mobilenetv2_bn_imagenet                 |      3.505 | (0.717, 0.904) | (0.79, 0.937)          | (0.585, 0.81)                | (0.678, 0.889)          | (0.738, 0.925)       | (0.185, 0.343)  | (0.016, 0.186) | (0.016, 0.049) |
| mobilenetv2_an_imagenet                 |      3.673 | (0.732, 0.912) | (0.803, 0.943)         | (0.605, 0.823)               | (0.695, 0.899)          | (0.756, 0.933)       | (0.204, 0.372)  | (0.028, 0.225) | (0.018, 0.05)  |
| densenet_121_an_imagenet                |      8.342 | (0.774, 0.936) | (0.837, 0.961)         | (0.651, 0.86)                | (0.742, 0.926)          | (0.792, 0.954)       | (0.249, 0.427)  | (0.085, 0.37)  | (0.014, 0.053) |

- Models trained with a normal training setup: 120 epochs, cosine learning rate scheduler, SGD+Momentum, etc. (MobileNet-v2 trained with 150 epochs). All models are trained with 8 Nvidia V100 GPUs.  See [Attentive Normalization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620069.pdf)(ECCV2020) for details.
- 'bn': BatchNorm (BN) is used.
- 'an': Attentive Normalization (AN) with BN backbone is used. 'an_bn1' means that 'an' is used to replace the first 'bn' in a building block (basic block or bottleneck block), and similarly for 'an_bn2', 'an_bn3' and 'an_bn12'. 'an_all' means that all 'bn' layers are replaced by 'an'.


# Top-1 and Top-5 Accuracy (w/ an advance training setup)
|                  Model                  | Params (M) |  imagenet-1k   | imagenet-1k-reassessed | imagenetv2-matched-frequency | imagenetv2-threshold0.7 | imagenetv2-topimages | imagenet-sketch |   imagenet-a   |   imagenet-o   |
|-----------------------------------------|-----------:|----------------|------------------------|------------------------------|-------------------------|----------------------|-----------------|----------------|----------------|
| aognet_12m_bn_imagenet_200e_ls_mixup    |     12.259 | (0.782, 0.943) | (0.85, 0.967)          | (0.666, 0.869)               | (0.752, 0.935)          | (0.8, 0.958)         | (0.286, 0.471)  | (0.086, 0.367) | (0.017, 0.053) |
| aognet_12m_an_imagenet_200e_ls_mixup    |     12.373 | (0.795, 0.947) | (0.857, 0.969)         | (0.681, 0.88)                | (0.767, 0.943)          | (0.815, 0.963)       | (0.285, 0.471)  | (0.133, 0.426) | (0.019, 0.055) |
| aognet_40m_bn_imagenet_200e_ls_mixup    |     40.153 | (0.812, 0.956) | (0.868, 0.972)         | (0.699, 0.891)               | (0.781, 0.948)          | (0.825, 0.967)       | (0.306, 0.489)  | (0.184, 0.496) | (0.017, 0.052) |
| aognet_40m_an_imagenet_200e_ls_mixup    |     40.389 | (0.818, 0.957) | (0.872, 0.973)         | (0.71, 0.898)                | (0.792, 0.951)          | (0.83, 0.968)        | (0.306, 0.49)   | (0.233, 0.549) | (0.017, 0.052) |
| resnetv1d_50_bn_imagenet_200e_ls_mixup  |     25.576 | (0.792, 0.945) | (0.851, 0.965)         | (0.669, 0.868)               | (0.758, 0.935)          | (0.805, 0.959)       | (0.271, 0.45)   | (0.088, 0.359) | (0.019, 0.055) |
| resnetv1d_50_an_imagenet_200e_ls_mixup  |     25.775 | (0.8, 0.95)    | (0.857, 0.969)         | (0.686, 0.882)               | (0.768, 0.937)          | (0.817, 0.961)       | (0.265, 0.441)  | (0.146, 0.432) | (0.016, 0.049) |
| resnetv1d_101_bn_imagenet_200e_ls_mixup |     44.568 | (0.802, 0.951) | (0.86, 0.969)          | (0.685, 0.882)               | (0.772, 0.945)          | (0.817, 0.967)       | (0.295, 0.477)  | (0.141, 0.43)  | (0.018, 0.051) |
| resnetv1d_101_an_imagenet_200e_ls_mixup |     45.020 | (0.81, 0.954)  | (0.863, 0.971)         | (0.696, 0.887)               | (0.778, 0.945)          | (0.82, 0.965)        | (0.3, 0.483)    | (0.199, 0.488) | (0.017, 0.049) |


# Top-1 and Top-5 Accuracy (AN as a strong alternative to the [Squeeze-Excitation (SE) module](https://arxiv.org/pdf/1709.01507.pdf))
|                  Model                  | Params (M) |  imagenet-1k   | imagenet-1k-reassessed | imagenetv2-matched-frequency | imagenetv2-threshold0.7 | imagenetv2-topimages | imagenet-sketch |   imagenet-a   |   imagenet-o   |
|-----------------------------------------|-----------:|----------------|------------------------|------------------------------|-------------------------|----------------------|-----------------|----------------|----------------|
| resnet_50_se_bn2_imagenet               |     26.186 | (0.779, 0.939) | (0.842, 0.961)         | (0.658, 0.867)               | (0.748, 0.933)          | (0.796, 0.957)       | (0.236, 0.409)  | (0.062, 0.326) | (0.018, 0.054) |
| resnet_50_an_bn2_imagenet               |     25.755 | (0.784, 0.941) | (0.844, 0.961)         | (0.666, 0.865)               | (0.748, 0.929)          | (0.798, 0.955)       | (0.247, 0.422)  | (0.082, 0.349) | (0.017, 0.053) |
| resnet_50_se_bn3_imagenet               |     28.072 | (0.777, 0.939) | (0.84, 0.961)          | (0.654, 0.863)               | (0.74, 0.928)           | (0.794, 0.954)       | (0.239, 0.411)  | (0.06, 0.318)  | (0.017, 0.057) |
| resnet_50_an_bn3_imagenet               |     26.349 | (0.782, 0.94)  | (0.839, 0.961)         | (0.659, 0.863)               | (0.746, 0.93)           | (0.796, 0.955)       | (0.246, 0.416)  | (0.086, 0.36)  | (0.014, 0.049) |
| resnet_50_se_bn123_imagenet             |     29.329 | (0.779, 0.94)  | (0.842, 0.962)         | (0.658, 0.863)               | (0.745, 0.926)          | (0.796, 0.952)       | (0.235, 0.405)  | (0.061, 0.328) | (0.018, 0.056) |
| resnet_50_an_all_imagenet               |     26.947 | (0.778, 0.938) | (0.837, 0.96)          | (0.657, 0.859)               | (0.743, 0.925)          | (0.794, 0.952)       | (0.236, 0.407)  | (0.084, 0.356) | (0.016, 0.048) |
