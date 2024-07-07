_base_ = [
    '../../yolox_base/yolox_x_8x8_300e_base.py',
]
ann_prefix2 = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/InOutDoorPeopleRGBD/coco_ann/'

train_ann_list = [
    ann_prefix2 + 'InOutDoorPeopleRGBD_train_set_correct.json',
]

img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/InOutDoorPeopleRGBD/Depth/',
]

val_list = [
    ann_prefix2 + 'InOutDoorPeopleRGBD_test_set_correct.json',
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/InOutDoorPeopleRGBD/Depth/',
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


dist_params = dict(backend='nccl')
