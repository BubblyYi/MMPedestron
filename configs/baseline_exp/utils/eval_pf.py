# @Time    : 10/10/2023 14:37
# @Author  : BubblyYi
# @FileName: eval_pf.py
# @Software: PyCharm
import logging

import numpy as np
import torch
from mmcv.ops.nms import batched_nms as mmcv_batched_nms
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCO, COCOeval
from torchvision.ops import nms  # BC-compat

# from torchvision.ops import boxes as box_ops


def xywh2xyxy(bboxes):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    return [[
        bbox[0],
        bbox[1],
        bbox[2] + bbox[0],
        bbox[3] + bbox[1],
    ] for bbox in bboxes]


def batched_nms(boxes, scores, idxs, iou_threshold):
    """Same as torchvision.ops.boxes.batched_nms, but safer."""
    assert boxes.shape[-1] == 4
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        dets, keep = mmcv_batched_nms(
            boxes, scores, idxs, dict(
                type='nms', iou_threshold=iou_threshold))
        return keep
        # return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def nms_1(bboex_all, scores_all, classes_all, iou_threshold=0.5):
    xyxy_bboex_all = xywh2xyxy(bboex_all)
    classes = torch.Tensor(classes_all)
    scores = torch.Tensor(scores_all)
    boxes = torch.Tensor(xyxy_bboex_all)
    bboex_all = torch.Tensor(bboex_all)
    # Perform nms
    keep_id = batched_nms(boxes, scores, classes, iou_threshold)
    # Add to output
    out_boxes = bboex_all[keep_id].numpy().tolist()
    out_scores = scores[keep_id].numpy().tolist()
    out_class = classes[keep_id].numpy().tolist()

    return out_boxes, out_scores, out_class


def prob_fusion(cocoGt, model1_pred_json_path='', model2_pred_json_path=''):
    img_ids = cocoGt.get_img_ids()
    print('start prob_fusion stage', flush=True)
    model1_pred = cocoGt.loadRes(model1_pred_json_path)
    model2_pred = cocoGt.loadRes(model2_pred_json_path)

    fusion_res = []

    for idx in img_ids:

        model1_pred_bbox = model1_pred.loadAnns(
            model1_pred.getAnnIds(imgIds=idx))
        model2_pred_bbox = model2_pred.loadAnns(
            model2_pred.getAnnIds(imgIds=idx))

        bboex_all = []
        scores_all = []
        classes_all = []
        for item in model1_pred_bbox:
            bboex_all.append(item['bbox'])
            classes_all.append(item['category_id'])
            scores_all.append(item['score'])

        for item2 in model2_pred_bbox:
            bboex_all.append(item2['bbox'])
            classes_all.append(item2['category_id'])
            scores_all.append(item2['score'])

        if len(bboex_all) > 0:
            out_boxes, out_scores, out_class = nms_1(
                bboex_all, scores_all, classes_all)
            for curr_bbox, curr_score in zip(out_boxes, out_scores):
                fusion_item = {}
                fusion_item['image_id'] = idx
                fusion_item['bbox'] = curr_bbox
                fusion_item['score'] = curr_score
                fusion_item['category_id'] = 1
                fusion_res.append(fusion_item)

    return fusion_res


def evaluate(fusion_res, cocoGt, proposal_nums=(100, 300, 1000)):
    print('start evaluate stage', flush=True)

    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    eval_results = {}
    metrics = ['bbox']
    logger = None
    metric_items = None
    for metric in metrics:
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        try:
            cocoDt = cocoGt.loadRes(fusion_res)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger,
                level=logging.ERROR)
            break

        iou_type = 'bbox' if metric == 'proposal' else metric
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cat_ids = cocoGt.get_cat_ids(cat_names=('person', ))
        img_ids = cocoGt.get_img_ids()
        cocoEval.params.catIds = cat_ids
        cocoEval.params.imgIds = img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                    'AR_m@1000', 'AR_l@1000'
                ]

            for item in metric_items:
                val = float(
                    f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
    return eval_results


if __name__ == '__main__':
    gt_prefix = '/mnt/lustre/share/evs/gt/'
    gt_list = ['annotations_person_align_v2_daytime_val_average_bbox.json',
               ]

    pred_prefix = '/mnt/lustre/share/evs/pred/'

    pred_list1 = ['faster_rcnn_person_fusion_pro_evs_v2_daytime_val.bbox.json',
                  ]

    pred_list2 = ['faster_rcnn_person_fusion_pro_rgb_v2_daytime_val.bbox.json',
                  ]

    for i in range(len(gt_list)):
        cocoGt = COCO(gt_prefix + gt_list[i])
        fusion_res = prob_fusion(
            cocoGt, pred_prefix + pred_list1[i],
            pred_prefix + pred_list2[i])
        print('gt_list:', gt_list[i])
        eval_results = evaluate(fusion_res, cocoGt)
