_base_ = [
    '../../co_dino_base/co_dino_5scale_r50_1x_base.py',
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
