# @Time    : 31/01/2024 15:41
# @Author  : BubblyYi
# @FileName: transforms.py
# @Software: PyCharm
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy import random


@PIPELINES.register_module()
class MaskImages(object):

    def __init__(self, masked_mod_list=[]):
        self.masked_mod_list = masked_mod_list

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        for key in self.masked_mod_list:
            results[key] = results[key] * 0
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(drop_p={self.drop_p})'
        return repr_str


@PIPELINES.register_module()
class RandomDropImages(object):

    def __init__(self, drop_p=0.1):
        if drop_p is not None:
            assert drop_p >= 0 and drop_p <= 1
        self.drop_p = drop_p

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        special_list = ['optical_flow_x', 'optical_flow_y']
        if np.random.rand() < self.drop_p:
            only_one_valid_key = random.choice(results['valid_img_fields'])
            if np.max(results[only_one_valid_key]) == 0:
                only_one_valid_key = 'img'
            is_do_special_drop = False
            if only_one_valid_key in special_list:
                is_do_special_drop = True
            for key in results['valid_img_fields']:
                if key == only_one_valid_key:
                    continue
                else:
                    if not is_do_special_drop:
                        results[key] = results[key] * 0
                    else:
                        if key not in special_list:
                            results[key] = results[key] * 0
            if is_do_special_drop:
                results['valid_img_fields'] = special_list
            else:
                results['valid_img_fields'] = [only_one_valid_key]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(drop_p={self.drop_p})'
        return repr_str
