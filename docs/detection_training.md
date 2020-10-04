# Training a Detector

It follows the [original MMDetection setup](../mmdetection/docs/geet_started.md).

## Change the ImageNet trained feature backbones
-  **All the ImageNet models are released on Google Drive** ([download link](https://drive.google.com/drive/folders/1puKc5g03bnt1qtzaHLCxu5-8tWmlo_WP?usp=sharing))
- Check the configuration files in the `configs_ivmcl/detection` folder.
- Change the `pretrained` path in the configuration file accordingly.

`E.g.`,

```shell
cd iVMCL-Release/mmdetection
./scripts_ivmcl/train_detection_dist.sh configs_ivmcl/detection/mask_rcnn_r50_an_conv_head_fpn.py
```
