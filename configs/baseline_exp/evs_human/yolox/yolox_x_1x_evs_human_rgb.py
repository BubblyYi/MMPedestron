_base_ = [
    '../../yolox_base/yolox_x_8x8_300e_base.py',
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

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        pipeline=[
            dict(type='LoadImageFromFile',
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
    ),
    test=dict(
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
    )
)
