import argparse

import torch

from mmcv import Config
from mmdet.models import build_backbone, build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check parameter keys')
    parser.add_argument('cfg_file', help='input cfg filename')
    parser.add_argument('ckpt_file', help='checkpoint filename')
    parser.add_argument('--is_detector',  action='store_true',
                        help='if the model is  a detector')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.cfg_file)
    if args.is_detector:
        model = build_detector(cfg.model,
                train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_backbone(cfg.model)

    ckpt = torch.load(args.ckpt_file, map_location='cpu')
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        if args.is_detector and '.fc.' in k:
            del state_dict[k]

    model_state_dict = model.state_dict()
    for k1, k2 in zip(model_state_dict, state_dict):
        sz1 = model_state_dict[k1].size()
        sz2 = state_dict[k2].size()
        print(
            f'{sz1==sz2} -- {k1}: {sz1} ------------ {k2[len("module.network."):]}: {sz2}')


if __name__ == '__main__':
    main()




