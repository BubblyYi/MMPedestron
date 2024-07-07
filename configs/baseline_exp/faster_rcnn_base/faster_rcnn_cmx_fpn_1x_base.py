_base_ = [
    'faster_rcnn_r50_fpn_1x_base.py',
]

pretrained = 'YOUR_DTAT_ROOT_PATH/mmpedestron_models/backbone/swin_small_patch4_window7_224_22k.pth'

model = dict(
    type='DualFasterRCNN',
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='DualMulitSwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        in_x_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer='LN',
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        )),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5),
)


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

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(
        type='LoadMultiModalitiesImages',
        mod_path_mapping_dict={
            'LLVIP': {
                'img': {
                    'org_key': 'LLVIP/visible',
                    'target_key': 'LLVIP/visible'},
                'ir_img': {
                    'org_key': 'LLVIP/visible',
                    'target_key': 'LLVIP/infrared'}}},
        mod_list=[
            'img',
            'ir_img',
        ],
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
        type='MulitAllImageFormatBundle',
        extra_image_list=[
            'ir_img',
        ]),
    dict(
        type='Collect',
        keys=[
            'img',
            'ir_img',
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
            'valid_img_fields',
            'img_fields')),
]

test_pipeline = [
    dict(type='LoadMultiModalitiesImages',
         mod_path_mapping_dict={
             'LLVIP': {
                 'img': {
                     'org_key': 'LLVIP/visible',
                     'target_key': 'LLVIP/visible'
                 },
                 'ir_img': {
                     'org_key': 'LLVIP/visible',
                     'target_key': 'LLVIP/infrared'
                 }
             }
         },
         mod_list=[
             'img',
             'ir_img',
         ],
         file_client_args=dict(backend='petrel')),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img', 'ir_img']),
             dict(type='Collect', keys=['img',
                                        'ir_img'],
                  meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                             'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                             'img_norm_cfg', 'valid_img_fields', 'img_fields')),
         ])
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
        pipeline=train_pipeline,
        classes=('person', )),
    val=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        pipeline=test_pipeline,
        classes=('person', )),
    test=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        pipeline=test_pipeline,
        classes=('person', )))
evaluation = dict(interval=1,
                  metric='bbox',
                  )

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
