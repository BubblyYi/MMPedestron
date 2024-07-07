# @Time    : 31/01/2024 15:44
# @Author  : BubblyYi
# @FileName: formatting.py
# @Software: PyCharm
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class MulitAllImageFormatBundle(DefaultFormatBundle):

    def __init__(self, extra_image_list=None):
        if extra_image_list is None:
            extra_image_list = []
        self.extra_image_list = extra_image_list

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for extra_image_key in self.extra_image_list:
            ex_img = results[extra_image_key]
            if len(ex_img.shape) < 3:
                ex_img = np.expand_dims(ex_img, -1)
            ex_img = np.ascontiguousarray(ex_img.transpose(2, 0, 1))
            results[extra_image_key] = DC(to_tensor(ex_img), stack=True)

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels'
        ]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(to_tensor(
                results['gt_semantic_seg'][None, ...]), stack=True)
        return results
