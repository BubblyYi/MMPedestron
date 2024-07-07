_base_ = [
    '../../faster_rcnn_base/faster_rcnn_r50_fpn_1x_base.py',
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
evaluation = dict(
    interval=1, metric='bbox',
    is_save_json=True,
    out_path='YOUR_DTAT_ROOT_PATH/temp_exp/multispectral_baseline_exp/prob_out/stcrowd_rgb/')

dataset_type = 'CocoPedestronDataset_Prob_Only'
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
dist_params = dict(backend='nccl')
