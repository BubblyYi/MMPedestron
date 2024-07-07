_base_ = [
    '../../yolox_base/yolox_x_8x8_300e_base.py',
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
img_scale = (640, 640)  # height, width


test_pipeline = [
    dict(type='LoadMultiModalitiesImages',
         mod_path_mapping_dict={
             'STCrowd': {
                 'img': {
                     'org_key': 'STCrowd_Raw/left',
                     'target_key': 'STCrowd_Raw/pcd_mask'
                 }
             }
         },
         mod_list=['img'],
         file_client_args=dict(backend='petrel')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        pipeline=[
            dict(type='LoadMultiModalitiesImages',
                 mod_path_mapping_dict={
                     'STCrowd': {
                         'img': {
                             'org_key': 'STCrowd_Raw/left',
                             'target_key': 'STCrowd_Raw/pcd_mask'
                         }
                     }
                 },
                 mod_list=['img'],
                 file_client_args=dict(backend='petrel')),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
        classes=('person', ),
    ),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        pipeline=test_pipeline
    )
)
