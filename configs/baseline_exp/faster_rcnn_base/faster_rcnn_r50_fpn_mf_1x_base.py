_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'YOUR_DTAT_ROOT_PATH/mmpedestron_models/backbone/resnet50-19c8e357.pth'

model = dict(
    type='MFFasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    backbone2=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[256 * 2, 512 * 2, 1024 * 2, 2048 * 2],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ))

)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
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
                   'extra_mod_img'],
         file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='MulitAllImageFormatBundle',
         extra_image_list=['extra_mod_img']),
    dict(type='Collect', keys=['img', 'extra_mod_img', 'gt_bboxes', 'gt_labels']),
]
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
            dict(type='ImageToTensor', keys=['img', 'extra_mod_img']),
            dict(type='Collect', keys=['img', 'extra_mod_img']),
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
    samples_per_gpu=2,
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

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
