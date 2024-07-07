_base_ = [
    '../../faster_rcnn_base/faster_rcnn_r50_fpn_mf_1x_base.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiModalitiesImages',
         mod_path_mapping_dict={
             'InOutDoorPeopleRGBD': {
                 'img': {
                     'org_key': 'InOutDoorPeopleRGBD/Images',
                     'target_key': 'InOutDoorPeopleRGBD/Images'
                 },
                 'extra_mod_img': {
                     'org_key': 'InOutDoorPeopleRGBD/Images',
                     'target_key': 'InOutDoorPeopleRGBD/Depth'
                 }
             },
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
             'InOutDoorPeopleRGBD': {
                 'img': {
                     'org_key': 'InOutDoorPeopleRGBD/Images',
                     'target_key': 'InOutDoorPeopleRGBD/Images'
                 },
                 'extra_mod_img': {
                     'org_key': 'InOutDoorPeopleRGBD/Images',
                     'target_key': 'InOutDoorPeopleRGBD/Depth'
                 }
             },
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


ann_prefix2 = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/InOutDoorPeopleRGBD/coco_ann/'

train_ann_list = [
    ann_prefix2 + 'InOutDoorPeopleRGBD_train_set_correct.json',
]

img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/InOutDoorPeopleRGBD/Images/',
]

val_list = [
    ann_prefix2 + 'InOutDoorPeopleRGBD_test_set_correct.json',
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/InOutDoorPeopleRGBD/Images/',
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
