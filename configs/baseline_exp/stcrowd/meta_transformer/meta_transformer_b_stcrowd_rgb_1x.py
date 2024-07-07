# @Time    : 22/09/2023 12:59
# @Author  : BubblyYi
# @FileName: meta_transformer_b_llvip_rgb_1x.py
# @Software: PyCharm
_base_ = [
    '../../meta_transformer_base/cascade_mask_rcnn_meta_transformer_adapter_base_fpn.py',
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

dataset_type = 'CocoPedestronDataset'
data = dict(
    train=dict(
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,),
    val=dict(
        ann_file=val_list,
        img_prefix=val_img_prefix_list
    ),
    test=dict(
        ann_file=val_list,
        img_prefix=val_img_prefix_list
    )
)
