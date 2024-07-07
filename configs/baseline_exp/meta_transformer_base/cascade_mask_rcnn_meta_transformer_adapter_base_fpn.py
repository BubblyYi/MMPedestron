# @Time    : 16/08/2023 16:11
# @Author  : BubblyYi
# @FileName: cascade_mask_rcnn_meta_transformer_adapter_base_fpn.py
# @Software: PyCharm
# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'YOUR_DTAT_ROOT_PATH/mmpedestron_models/backbone/Image_Meta-Transformer_base_patch16.pth'
model = dict(
    backbone=dict(
        _delete_=True, type='ViTAdapter', patch_size=16, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.3,
        conv_inplane=64, n_points=4, deform_num_heads=12, cffn_ratio=0.25,
        deform_ratio=0.5, interaction_indexes=[[0, 2],
                                               [3, 5],
                                               [6, 8],
                                               [9, 11]],
        window_attn=[True, True, False, True, True, False, True, True, False,
                     True, True, False],
        window_size=[14, 14, None, 14, 14, None, 14, 14, None, 14, 14, None],
        pretrained=pretrained),
    neck=dict(
        type='FPN', in_channels=[768, 768, 768, 768],
        out_channels=256, num_outs=5),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead', num_shared_convs=4,
                num_shared_fcs=1, in_channels=256,
                conv_out_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False, reg_decoded_bbox=True,
                norm_cfg=dict(
                    type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead', num_shared_convs=4,
                num_shared_fcs=1, in_channels=256,
                conv_out_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False, reg_decoded_bbox=True,
                norm_cfg=dict(
                    type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead', num_shared_convs=4,
                num_shared_fcs=1, in_channels=256,
                conv_out_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False, reg_decoded_bbox=True,
                norm_cfg=dict(
                    type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='GIoULoss', loss_weight=10.0))],
        mask_roi_extractor=None, mask_head=None),)

# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(
        type='LoadImageFromFile', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333),
                           (512, 1333),
                           (544, 1333),
                           (576, 1333),
                           (608, 1333),
                           (640, 1333),
                           (672, 1333),
                           (704, 1333),
                           (736, 1333),
                           (768, 1333),
                           (800, 1333)],
                multiscale_mode='value', keep_ratio=True)],
            [
            dict(
                type='Resize',
                img_scale=[(400, 1333),
                           (500, 1333),
                           (600, 1333)],
                multiscale_mode='value', keep_ratio=True),
            dict(
                type='RandomCrop', crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
            dict(
                type='Resize',
                img_scale=[(480, 1333),
                           (512, 1333),
                           (544, 1333),
                           (576, 1333),
                           (608, 1333),
                           (640, 1333),
                           (672, 1333),
                           (704, 1333),
                           (736, 1333),
                           (768, 1333),
                           (800, 1333)],
                multiscale_mode='value', override=True,
                keep_ratio=True)]]),
    dict(
        type='RandomCrop', crop_type='absolute_range', crop_size=(1024, 1024),
        allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile',
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

evaluation = dict(
    interval=1,
    metric='bbox',


)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'level_embed': dict(decay_mult=0.),
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))


optimizer_config = dict(grad_clip=None)
total_epochs = 12
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/share/'
find_unused_parameters = True
