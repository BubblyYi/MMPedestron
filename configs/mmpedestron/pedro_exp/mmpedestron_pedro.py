# model settings
num_dec_layer = 6
lambda_2 = 2.0
pretrained = 'YOUR_DTAT_ROOT_PATH/mmpedestron_models/backbone/dualvit_s_384.pth.tar'

model = dict(
    type='MMPedestron',
    backbone=dict(
        type='UNIXVit',
        stem_width=32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14],
        mlp_ratios=[8, 8, 4, 3],
        norm_layer='LN',
        depths=[3, 4, 6, 3],
        drop_path_rate=0.3,
        score_embed_nums=1,
        num_scores=2,
        mod_nums=1,
        with_cp=True,
        pretrained=pretrained
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[64, 128, 320, 448],
        out_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * num_dec_layer * lambda_2),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0 * num_dec_layer * lambda_2)),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=1,
        num_feature_levels=5,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        mixed_selection=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)),
        transformer=dict(
            type='CoDinoTransformer',
            with_pos_coord=True,
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,  # number of layers that use checkpoint
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256, num_levels=5, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
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
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0 * num_dec_layer * lambda_2),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0 * num_dec_layer * lambda_2)))],
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0 * num_dec_layer * lambda_2),
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=2.0 *
            num_dec_layer *
            lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * num_dec_layer * lambda_2))],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost',
                    weight=5.0,
                    box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
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
                nms_pre=4000,
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
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)],
    test_cfg=[
        dict(
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=1000)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=1000),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
max_img_size = 1333

train_pipeline = [
    dict(type='LoadMultiModalitiesImages',
         mod_path_mapping_dict={
             'PEDRo_events_dataset': {
                 'img': {
                     'org_key': 'PEDRo_events_dataset/',
                     'target_key': 'PEDRo_events_dataset/'
                 },
                 'extra_mod_img': {
                     'org_key': 'PEDRo_events_dataset/',
                     'target_key': 'PEDRo_events_dataset/'
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
                    img_scale=[(480, max_img_size), (512, max_img_size), (544, max_img_size),
                               (576, max_img_size), (608,
                                                     max_img_size), (640, max_img_size),
                               (672, max_img_size), (704,
                                                     max_img_size), (736, max_img_size),
                               (768, max_img_size), (800, max_img_size)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[
                        (400, max_img_size),
                        (500, max_img_size),
                        (600, max_img_size)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, max_img_size), (512, max_img_size), (544, max_img_size),
                               (576, max_img_size), (608,
                                                     max_img_size), (640, max_img_size),
                               (672, max_img_size), (704,
                                                     max_img_size), (736, max_img_size),
                               (768, max_img_size), (800, max_img_size)],
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

test_pipeline = [
    dict(type='LoadMultiModalitiesImages',
         mod_path_mapping_dict={
             'PEDRo_events_dataset': {
                 'img': {
                     'org_key': 'PEDRo_events_dataset/',
                     'target_key': 'PEDRo_events_dataset/'
                 },
                 'extra_mod_img': {
                     'org_key': 'PEDRo_events_dataset/',
                     'target_key': 'PEDRo_events_dataset/'
                 }
             }
         },
         mod_list=['img',
                   'extra_mod_img', ],
         file_client_args=dict(backend='petrel')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(max_img_size, 800),
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
ann_prefix = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/PEDRo_events_dataset/coco_ann/'

train_ann_list = [
    ann_prefix + 'pedro_train.json',

]
img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/PEDRo_events_dataset/train_img/',
]

val_list = [
    ann_prefix + 'pedro_val.json',
    ann_prefix + 'pedro_test.json'
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/PEDRo_events_dataset/val_img/',
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/PEDRo_events_dataset/test_img/',
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
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

optimizer_config = dict(type='GC_OptimizerHook',
                        grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 10])
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
load_from = 'YOUR_DTAT_ROOT_PATH/mmpedestron_models/mmpedestron_mix5datasets_best/mmpedestron_mix5datasets_best-d0d55169.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
