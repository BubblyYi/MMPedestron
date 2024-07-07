_base_ = [
    '../../mmpedestron_base/faster_rcnn_mmpedestron_1x_faster_rcnn_base.py',
]
ann_prefix = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/evs_dataset/evs_human_1013/'

train_ann_list = [
    ann_prefix + 'train.json',

]

img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/evs_human_1013/train/rgb/',
]

val_list = [
    ann_prefix + 'val.json',
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/evs_human_1013/val/rgb/',

]
model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[1.0, 1.5, 2.0, 2.5, 3.0],
            strides=[4, 8, 16, 32, 64]),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=0.5),
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
                max_per_img=2000,
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
                nms_post=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.01,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=1000)
            # soft-nms is also supported for rcnn testing
            # e.g., nms=dict(type='soft_nms', iou_threshold=0.5,
            # min_score=0.05)
        ))
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

train_pipeline = [
    dict(
        type='LoadMultiModalitiesImages',
        mod_path_mapping_dict={
            'evs_human_1013': {
                'img': {
                    'org_key': 'evs_human_1013/train/rgb',
                    'target_key': 'evs_human_1013/train/rgb'},
                'extra_mod_img': {
                    'org_key': 'evs_human_1013/train/rgb',
                    'target_key': 'evs_human_1013/train/evs'}}},
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
             'evs_human_1013': {
                 'img': {
                     'org_key': 'evs_human_1013/val/rgb',
                     'target_key': 'evs_human_1013/val/rgb'
                 },
                 'extra_mod_img': {
                     'org_key': 'evs_human_1013/val/rgb',
                     'target_key': 'evs_human_1013/val/evs'
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

dataset_type = 'CocoPedestronDataset'
data = dict(samples_per_gpu=1,
            workers_per_gpu=2,
            train=dict(type=dataset_type,
                       filter_empty_gt=False,
                       ann_file=train_ann_list,
                       img_prefix=img_prefix_list,
                       pipeline=train_pipeline,
                       classes=('person', )),
            val=dict(type=dataset_type,
                     ann_file=val_list,
                     img_prefix=val_img_prefix_list,
                     pipeline=test_pipeline,
                     classes=('person', )),
            test=dict(type=dataset_type,
                      ann_file=val_list,
                      img_prefix=val_img_prefix_list,
                      pipeline=test_pipeline,
                      classes=('person', )))
evaluation = dict(
    interval=1,
    metric='bbox',

)

log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])
