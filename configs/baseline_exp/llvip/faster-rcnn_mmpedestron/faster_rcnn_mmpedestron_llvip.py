_base_ = [
    '../../mmpedestron_base/faster_rcnn_mmpedestron_1x_faster_rcnn_base.py',
]

model = dict(
    backbone=dict(
        drop_path_rate=0.3,
    )
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

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
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomDropImages', drop_p=0.3),
    dict(type='MulitAllImageFormatBundle',
         extra_image_list=['extra_mod_img']),
    dict(type='Collect', keys=['img',
                               'extra_mod_img',
                               'gt_bboxes',
                               'gt_labels',
                               'gt_bboxes_ignore'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                                                               'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                                               'flip_direction', 'img_norm_cfg', 'valid_img_fields'))
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
                       pipeline=train_pipeline,
                       classes=('person', )),
            val=dict(type=dataset_type,
                     pipeline=test_pipeline,
                     classes=('person', )),
            test=dict(type=dataset_type,
                      pipeline=test_pipeline,
                      classes=('person', )))
