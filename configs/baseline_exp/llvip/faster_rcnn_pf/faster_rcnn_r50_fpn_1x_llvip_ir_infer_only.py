_base_ = [
    '../../faster_rcnn_base/faster_rcnn_r50_fpn_1x_base.py',
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

evaluation = dict(
    interval=1, metric='bbox',
    is_save_json=True,
    out_path='YOUR_DTAT_ROOT_PATH/temp_exp/multispectral_baseline_exp/prob_out/llvip_ir/')


dataset_type = 'CocoPedestronDataset_Prob_Only'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=train_ann_list,
        img_prefix=img_prefix_list,
        classes=('person', )),

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
    )
)
