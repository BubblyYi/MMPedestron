# @Time    : 31/01/2024 15:48
# @Author  : BubblyYi
# @FileName: coco_pedestron.py
# @Software: PyCharm
import copy
import itertools
import json
import logging
import os

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from terminaltables import AsciiTable

from mmdet_custom.core.evaluation.crowdhuman_eval_tools import Database


def make_folder(folder_path):
    """make dir.

    :param folder_path: dir path
    :return: none
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    print('make dir: %s done' % folder_path)


def save_json(list_file, path):
    with open(path, 'w') as f:
        json.dump(list_file,
                  f,
                  indent=1,
                  ensure_ascii=False)
    print('save json done:', path, flush=True)


@DATASETS.register_module()
class CocoPedestronDataset(CocoDataset):
    CLASSES = ('person', )

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if w * h <= 0:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore,
                   masks=gt_masks_ann,
                   seg_map=seg_map)

        return ann

    def coco_format_converter(self, bbox_res):
        new_res = []
        for idx in range(len(self.data_infos)):
            img_info = self.data_infos[idx]
            curr_id = img_info['filename'].replace('Images/', '')
            curr_id = curr_id.replace('.jpg', '')
            # print('curr_id:', curr_id)
            det_bbox = []
            curr_v_bboxes = bbox_res[idx][0]
            # print('curr_v_bboxes:', curr_v_bboxes)
            for bbox in curr_v_bboxes:
                bbox_item = {
                    'box': self.xyxy2xywh(bbox[:4]),
                    'tag': 1,
                    'score': np.float(bbox[4])
                }
                det_bbox.append(bbox_item)
            item = {
                'ID': curr_id,
                'height': img_info['height'],
                'width': img_info['width'],
                'dtboxes': det_bbox
            }
            new_res.append(item)
        return new_res

    def compute_crowdhuman_AP(self,
                              gt_path,
                              dt_path,
                              target_key='box',
                              mode=0):
        database = Database(gt_path, dt_path, target_key, None, mode)
        database.compare()
        mAP, _ = database.eval_AP()
        MR, _ = database.eval_MR()
        return mAP, MR

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        visible_body_results = copy.deepcopy(results)

        eval_results = {}
        tmp_dir = None
        crowdhuman_odgt_val_gt_path = kwargs.get(
            'crowdhuman_odgt_val_gt_path', None)

        if 'crowdhuman_coco' in self.ann_file and crowdhuman_odgt_val_gt_path is not None:
            converted_full_res = self.coco_format_converter(
                visible_body_results)
            crowdhuman_full_mAP, crowdhuman_full_MR = self.compute_crowdhuman_AP(
                crowdhuman_odgt_val_gt_path, converted_full_res, mode=0)
            eval_results['full_crowdhuman_mAP'] = crowdhuman_full_mAP
            eval_results['crowdhuman_full_MR'] = crowdhuman_full_MR
            print('-' * 60)
            print('full_crowdhuman_mAP:', crowdhuman_full_mAP)
            print('crowdhuman_full_MR:', crowdhuman_full_MR)
            print('-' * 60)
        else:
            metrics = metric if isinstance(metric, list) else [metric]
            allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
            for metric in metrics:
                if metric not in allowed_metrics:
                    raise KeyError(f'metric {metric} is not supported')
            if iou_thrs is None:
                iou_thrs = np.linspace(.5,
                                       0.95,
                                       int(np.round((0.95 - .5) / .05)) + 1,
                                       endpoint=True)
            if metric_items is not None:
                if not isinstance(metric_items, list):
                    metric_items = [metric_items]
            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix)

            cocoGt = self.coco
            for metric in metrics:
                msg = f'Evaluating {metric}...'
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                if metric == 'proposal_fast':
                    ar = self.fast_eval_recall(results,
                                               proposal_nums,
                                               iou_thrs,
                                               logger='silent')
                    log_msg = []
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
                        log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                    log_msg = ''.join(log_msg)
                    print_log(log_msg, logger=logger)
                    continue

                if metric not in result_files:
                    raise KeyError(f'{metric} is not in results')
                try:
                    cocoDt = cocoGt.loadRes(result_files[metric])
                except IndexError:
                    print_log(
                        'The testing results of the whole dataset is empty.',
                        logger=logger,
                        level=logging.ERROR)
                    break

                iou_type = 'bbox' if metric == 'proposal' else metric
                cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                cocoEval.params.catIds = self.cat_ids
                cocoEval.params.imgIds = self.img_ids
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
                                f'metric item {metric_item} is not supported'
                            )

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
                    if classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = cocoEval.eval['precision']
                        # precision: (iou, recall, cls, area range, max dets)
                        assert len(self.cat_ids) == precisions.shape[2]

                        results_per_category = []
                        for idx, catId in enumerate(self.cat_ids):
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = self.coco.loadCats(catId)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            results_per_category.append(
                                (f'{nm["name"]}', f'{float(ap):0.3f}'))

                        num_columns = min(6, len(results_per_category) * 2)
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = ['category', 'AP'] * (num_columns // 2)
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        print_log('\n' + table.table, logger=logger)

                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m',
                            'mAP_l'
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
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


@DATASETS.register_module()
class CocoPedestronDataset_FLIR(CocoPedestronDataset):
    CLASSES = ('person', 'bicycle', 'car')

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        visible_body_results = copy.deepcopy(results)

        eval_results = {}
        tmp_dir = None
        crowdhuman_odgt_val_gt_path = kwargs.get(
            'crowdhuman_odgt_val_gt_path', None)

        if 'crowdhuman_coco' in self.ann_file and crowdhuman_odgt_val_gt_path is not None:
            converted_full_res = self.coco_format_converter(
                visible_body_results)
            crowdhuman_full_mAP, crowdhuman_full_MR = self.compute_crowdhuman_AP(
                crowdhuman_odgt_val_gt_path, converted_full_res, mode=0)
            eval_results['full_crowdhuman_mAP'] = crowdhuman_full_mAP
            eval_results['crowdhuman_full_MR'] = crowdhuman_full_MR
            print('-' * 60)
            print('full_crowdhuman_mAP:', crowdhuman_full_mAP)
            print('crowdhuman_full_MR:', crowdhuman_full_MR)
            print('-' * 60)
        else:
            metrics = metric if isinstance(metric, list) else [metric]
            allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
            for metric in metrics:
                if metric not in allowed_metrics:
                    raise KeyError(f'metric {metric} is not supported')
            if iou_thrs is None:
                iou_thrs = np.linspace(.5, 0.95, int(
                    np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            if metric_items is not None:
                if not isinstance(metric_items, list):
                    metric_items = [metric_items]
            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix)

            cocoGt = self.coco
            for metric in metrics:
                msg = f'Evaluating {metric}...'
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                if metric == 'proposal_fast':
                    ar = self.fast_eval_recall(
                        results, proposal_nums, iou_thrs, logger='silent')
                    log_msg = []
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
                        log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                    log_msg = ''.join(log_msg)
                    print_log(log_msg, logger=logger)
                    continue

                if metric not in result_files:
                    raise KeyError(f'{metric} is not in results')
                try:
                    cocoDt = cocoGt.loadRes(result_files[metric])
                except IndexError:
                    print_log(
                        'The testing results of the whole dataset is empty.',
                        logger=logger,
                        level=logging.ERROR)
                    break

                iou_type = 'bbox' if metric == 'proposal' else metric
                cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                cocoEval.params.catIds = self.cat_ids
                cocoEval.params.imgIds = self.img_ids
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
                    if classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = cocoEval.eval['precision']
                        # precision: (iou, recall, cls, area range, max dets)
                        assert len(self.cat_ids) == precisions.shape[2]

                        results_per_category = []
                        for idx, catId in enumerate(self.cat_ids):
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = self.coco.loadCats(catId)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            results_per_category.append(
                                (f'{nm["name"]}', f'{float(ap):0.3f}'))

                        num_columns = min(6, len(results_per_category) * 2)
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = ['category', 'AP'] * (num_columns // 2)
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        print_log('\n' + table.table, logger=logger)

                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

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
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


@DATASETS.register_module()
class CocoPedestronDataset_Prob_Only(CocoPedestronDataset):

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return json_results

    def save_pred_json(self, results, out_path='', score_thr=0.01):
        """Convert detection results to COCO json style."""
        make_folder(out_path)
        out_name = os.path.join(out_path, 'pred_det_res.json')
        save_json(results, out_name)
        print('save json done:', out_name, flush=True)
        return None

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 **kwargs):
        out_path = kwargs.get('out_path', '')
        score_thr = kwargs.get('score_thr', 0.05)
        is_save_json = kwargs.get('is_save_json', False)
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if is_save_json:
            self.save_pred_json(result_files,
                                out_path=out_path,
                                score_thr=score_thr)
        eval_results = {}
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
