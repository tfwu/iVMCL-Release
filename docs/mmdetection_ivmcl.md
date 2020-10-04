# Modifications of MMDetection

The code is maintained to be almost the same as the remote master. When a new module is added, we adopt the naive way of adding a separate file, rather than modifying codes in existing files. By doing this, we will only need to update `__init__.py` files.

- [x] Add a few new backbones in
    ```
    iVMCL-Release
    ├── mmdetectoin
        ├── mmdet
            ├── models
                ├── backbones
                    ├── aognet.py
                    ├── resnet_an.py    # Attentive Normalization
                    ├── resnext_an.py   # Attentive Normalization
                    ├── densenet.py     # Attentive Normalization
                    ├── mobilenet_v2.py # Attentive Normalization
    ```

- [x] Add a new subfolder `ivmcl` consisting of modules for [AOGNets](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_AOGNets_Compositional_Grammatical_Architectures_for_Deep_Learning_CVPR_2019_paper.pdf) (CVPR2019) and for training ImageNet classifiers
    ```
    iVMCL-Release
    ├── mmdetectoin
        ├── mmdet
            ├── ivmcl
                ├── aog
                ├── data
                ├── loss
                ├── ops
                ├── optim
                ├── scheduler
                ├── utils
    ```

- [x] Add a new subfolder `configs_ivmcl` consisting of configuration files for classification and detection
    ```
    iVMCL-Release
    ├── mmdetectoin
        ├── configs_ivmcl
            ├── classification
            ├── detection       # slightly modified from original files in mmdetection/configs
    ```

- [x] Add a new script subfolder `scripts_ivmcl` and a tool subfolder `tools_ivmcl`
    ```
    iVMCL-Release
    ├── mmdetectoin
        ├── scripts_ivmcl
        ├── tools_ivmcl
    ```
