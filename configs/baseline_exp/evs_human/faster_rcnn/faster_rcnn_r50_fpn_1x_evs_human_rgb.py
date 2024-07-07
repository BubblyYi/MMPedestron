_base_ = [
    '../../faster_rcnn_base/faster_rcnn_r50_fpn_1x_base.py',
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

dataset_type = 'CocoPedestronDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        filter_empty_gt=False,
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        classes=('person', )
    ),
    val=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        classes=('person', )
    ),
    test=dict(
        type=dataset_type,
        ann_file=val_list,
        img_prefix=val_img_prefix_list,
        classes=('person', )
    ))

evaluation = dict(interval=1,
                  metric='bbox',
                  )
