# model settings
model = dict(
    type='ResNetAN',
    depth=50,
    num_stages=4,
    out_indices=(3,),
    norm_eval=False,
    num_classes=1000)
# dataset settings
data_root = 'data/ILSVRC2015/Data/CLS-LOC/'
data_cfg = dict(
    train_cfg = dict(
        type='NULL',
        crop_size=224,
        crop_min_scale=0.08,
        interpolation='bilinear',
        drop_last=False,
        mix_up_rate=0.0,
        label_smoothing_rate=0.0,
        num_classes=1000),
    val_cfg = dict(
        type='NULL',
        crop_size=224,
        crop_padding=32,
        interpolation='bilinear'))
# optimizer
optimizer = dict(
    opt='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=False,
    remove_norm_weigth_decay=False)
# learning policy
lr_config = dict(
    policy='cosine',
    warmup_epoch=5,
    warmup_multiplier=100,
    step_decay_rate=0.1,
    step_decay_epochs=[30,60,90])
# yapf:enable
total_epochs = 120
batch_size = 256  # per gpu
test_batch_size = 200
num_workers = 8
autoscale_lr_factor = 256. # autoscale lr
dist_params = dict(backend='nccl', init_method='env://')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
