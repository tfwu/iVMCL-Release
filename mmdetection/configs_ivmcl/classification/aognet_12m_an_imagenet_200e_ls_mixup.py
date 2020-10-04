# model settings
model = dict(
    type='AOGNet',
    aog_cfg=dict(
        dims=(2, 2, 4, 4),
        max_splits=(2, 2, 2, 2),
        extra_node_hierarchy=('4', '4', '4', '4'),
        remove_symmetric_children_of_or_node=(1, 2, 1, 2)
    ),
    stem_type='DeepStem',
    block='AOGBlock',
    block_num=(2, 2, 2, 1),
    filter_list=(32, 128, 256, 512, 824),
    ops_t_node=('Bottleneck', 'Bottleneck', 'Bottleneck', 'Bottleneck'),
    ops_and_node=('Bottleneck', 'Bottleneck', 'Bottleneck', 'Bottleneck'),
    ops_or_node=('Bottleneck', 'Bottleneck', 'Bottleneck', 'Bottleneck'),
    bn_ratios=(0.25, 0.25, 0.25, 0.25),
    t_node_no_slice=(False, False, False, False),
    t_node_handle_dblcnt=(False, False, False, False),
    non_t_node_handle_dblcnt=(True, True, True, True),
    or_node_reduction='sum',
    drop_rates=(0., 0., 0.1, 0.1),
    strides=(1, 2, 2, 2),
    dilations=(1, 1, 1, 1),
    with_group_conv=(0, 0, 0, 0),
    base_width=(4, 4, 4, 4),
    out_indices=(3,),
    norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
    num_affine_trans=(10, 10, 20, 20),
    norm_eval=False,
    zero_init_residual=False,
    use_extra_norm_ac_for_block=True,
    handle_dbl_cnt_in_weight_init=False,
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
        mix_up_rate=0.2,
        label_smoothing_rate=0.1,
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
