# @Time    : 22/09/2023 12:59
# @Author  : BubblyYi
# @FileName: meta_transformer_b_llvip_rgb_1x.py
# @Software: PyCharm
_base_ = [
    '../../meta_transformer_base/cascade_mask_rcnn_meta_transformer_adapter_base_fpn.py',
]

ann_prefix = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/LLVIP/ann_coco/'
train_ann_list = [
    ann_prefix + 'LLVIP_coco_train_change_cat_id.json',
]

img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/LLVIP/infrared/train/',
]

val_list = [
    ann_prefix + 'LLVIP_coco_val_change_cat_id.json',
]

val_img_prefix_list = [
    'YOUR_DTAT_ROOT_PATH/mmpedestron_images/LLVIP/infrared/test/',
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
