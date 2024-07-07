_base_ = [
    '../../mmpedestron_base/faster_rcnn_mmpedestron_1x_faster_rcnn_base.py',
]

ann_prefix4 = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/STCrowd_Raw/coco_ann/'

train_ann_list = [
    ann_prefix4 + 'train_set_lidar_mask.json'
]

img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/STCrowd_Raw/',

]

val_list = [
    ann_prefix4 + 'val_set_lidar_mask.json'
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/STCrowd_Raw/',
]


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

train_pipeline = [
    dict(
        type='LoadMultiModalitiesImages',
        mod_path_mapping_dict={
            'STCrowd': {
                'img': {
                    'org_key': 'STCrowd_Raw/left',
                    'target_key': 'STCrowd_Raw/left'},
                'extra_mod_img': {
                    'org_key': 'STCrowd_Raw/left',
                    'target_key': 'STCrowd_Raw/pcd_mask'}}},
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
             'STCrowd': {
                 'img': {
                     'org_key': 'STCrowd_Raw/left',
                     'target_key': 'STCrowd_Raw/left'
                 },
                 'extra_mod_img': {
                     'org_key': 'STCrowd_Raw/left',
                     'target_key': 'STCrowd_Raw/pcd_mask'
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
