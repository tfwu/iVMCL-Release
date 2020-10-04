_base_ = [
    './_base_/mask_rcnn_r50_an_fpn.py',
    '../../configs/_base_/datasets/coco_instance.py',
    '../../configs/_base_/schedules/schedule_2x.py'
]

# runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/xli47/Src/AttentiveNorm_Detection/pretrained_models/mask_rcnn_r50_an_conv_head_fpn_2x.pth'
resume_from = None
workflow = [('train', 1)]
