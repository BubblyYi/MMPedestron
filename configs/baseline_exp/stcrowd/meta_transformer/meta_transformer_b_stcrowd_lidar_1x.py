# @Time    : 22/09/2023 12:59
# @Author  : BubblyYi
# @FileName: meta_transformer_b_llvip_rgb_1x.py
# @Software: PyCharm
_base_ = [
    '../../meta_transformer_base/cascade_mask_rcnn_meta_transformer_adapter_base_fpn.py',
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

# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

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
        img_scale=(1333, 800),
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

dataset_type = 'CocoPedestronDataset'
data = dict(
    train=dict(
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        pipeline=train_pipeline
    ),
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
