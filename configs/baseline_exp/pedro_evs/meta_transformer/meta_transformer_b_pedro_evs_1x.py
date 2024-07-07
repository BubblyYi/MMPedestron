# @Time    : 22/09/2023 12:59
# @Author  : BubblyYi
# @FileName: meta_transformer_b_llvip_rgb_1x.py
# @Software: PyCharm
_base_ = [
    '../../meta_transformer_base/cascade_mask_rcnn_meta_transformer_adapter_base_fpn.py',
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
