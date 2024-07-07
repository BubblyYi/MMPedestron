# model settings
pretrained = 'YOUR_DTAT_ROOT_PATH/mmpedestron_models/backbone/dualvit_s_384.pth.tar'
model = dict(
    type='MulitFastRCNN',
    backbone=dict(
        type='UNIXVit',
        stem_width=32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14],
        mlp_ratios=[8, 8, 4, 3],
        norm_layer='LN',
        depths=[3, 4, 6, 3],
        drop_path_rate=0.15,
        score_embed_nums=1,
        num_scores=2,
        mod_nums=1,
        with_cp=True,
        pretrained=pretrained
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 448],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(
        type='LoadMultiModalitiesImages',
        mod_path_mapping_dict={
            'LLVIP': {
                'img': {
                    'org_key': 'LLVIP/visible',
                    'target_key': 'LLVIP/visible'},
                'extra_mod_img': {
                    'org_key': 'LLVIP/visible',
                    'target_key': 'LLVIP/infrared'}}},
        mod_list=[
            'img',
            'extra_mod_img'],
        file_client_args=dict(
            backend='petrel')),
    dict(
        type='LoadAnnotations',
        with_bbox=True),
    dict(
        type='Resize',
        img_scale=(
            1333,
            800),
        keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.5),
    dict(
        type='Normalize',
        **img_norm_cfg),
    dict(
        type='Pad',
        size_divisor=32),
    dict(
        type='RandomDropImages',
        drop_p=0.3),
    dict(
        type='MulitAllImageFormatBundle',
        extra_image_list=['extra_mod_img']),
    dict(
        type='Collect',
        keys=[
            'img',
            'extra_mod_img',
            'gt_bboxes',
            'gt_labels',
            'gt_bboxes_ignore'],
        meta_keys=(
            'filename',
            'ori_filename',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'img_norm_cfg',
            'valid_img_fields'))]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.

test_pipeline = [
    dict(type='LoadMultiModalitiesImages',
         mod_path_mapping_dict={
             'LLVIP': {
                 'img': {
                     'org_key': 'LLVIP/visible',
                     'target_key': 'LLVIP/visible'
                 },
                 'extra_mod_img': {
                     'org_key': 'LLVIP/visible',
                     'target_key': 'LLVIP/infrared'
                 }
             }
         },
         mod_list=['img',
                   'extra_mod_img', ],
         file_client_args=dict(backend='petrel')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='MaskImages', masked_mod_list=['img']),
            dict(type='ImageToTensor', keys=['img',
                                             'extra_mod_img',
                                             ]),
            dict(type='Collect', keys=['img',
                                       'extra_mod_img',
                                       ]),
        ])
]

ann_prefix = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/LLVIP/ann_coco/'

train_ann_list = [
    ann_prefix + 'LLVIP_coco_train_change_cat_id.json',
]

img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/LLVIP/visible/train/',
]

val_list = [
    ann_prefix + 'LLVIP_coco_val_change_cat_id.json',
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/LLVIP/visible/test/',
]

dataset_type = 'CocoPedestronDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        filter_empty_gt=False,
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        classes=('person', ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        classes=('person', ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        classes=('person', ),
        pipeline=test_pipeline))
evaluation = dict(interval=1,
                  metric='bbox',
                  )
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
)

optimizer_config = dict(type='GC_OptimizerHook',
                        grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
