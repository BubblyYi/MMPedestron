_base_ = [
    '../../co_dino_base/co_dino_5scale_r50_1x_base.py',
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

dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        filter_empty_gt=False,
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        classes=('person', ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        classes=('person', ),
    ),
    test=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        classes=('person', ),
    ))
