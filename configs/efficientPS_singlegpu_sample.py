model = dict(
    type='EfficientPS',
    pretrained=True,
    backbone=dict(
        type='tf_efficientnet_b5', 
        act_cfg = dict(type="Identity"),  
        norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='TWOWAYFPN',
        in_channels=[40, 64, 176, 2048], #b0[24, 40, 112, 1280], #b4[32, 56, 160, 1792], #b5[40, 64, 176, 2048]
        out_channels=256,
        norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
        act_cfg=None,
        num_outs=4),

    semantic_head=dict(
        type='EfficientPSSemanticHead',
        in_channels=256,
        conv_out_channels=128,
        num_classes=19,
        nclasses=6,
        ignore_label=255,
        loss_weight=1.0,
        ohem=0.25,
        norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
        act_cfg=None))

train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False)
    )
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.5,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5),
    panoptic=dict(
        overlap_thr=0.5,
        min_stuff_area=2048))

dataset_type = 'ForestDataset'
data_root = 'data/freiburg/'
img_norm_cfg = dict(
    mean=[106.433, 116.617, 119.559], std=[65.496, 67.6, 74.123], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=False, with_seg=True),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 2048)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img',  'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
         seg_prefix=data_root + 'stuffthingmaps/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        seg_prefix=data_root + 'stuffthingmaps/val/',
        panoptic_gt=data_root + 'freiburg_panoptic_val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        seg_prefix=data_root + 'stuffthingmaps/val/',
        panoptic_gt=data_root + 'freiburg_panoptic_val',
        pipeline=test_pipeline))
evaluation = dict(interval=10, metric=['segm'])

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[120, 144]) 
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = "./efficientPS_cityscapes/model/model.pth"
# load_from="./work_dirs/checkpoints/epoch_42.pth"
resume_from = None
workflow = [('train', 1)]
