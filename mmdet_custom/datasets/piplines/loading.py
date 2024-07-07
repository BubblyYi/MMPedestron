# @Time    : 31/01/2024 15:36
# @Author  : BubblyYi
# @FileName: loading.py
# @Software: PyCharm
import os.path as osp
import sys
from io import BytesIO

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile


@PIPELINES.register_module()
class LoadMultiModalitiesImages(LoadImageFromFile):

    def __init__(
            self,
            mod_path_mapping_dict={
                'LLVIP': {
                    'img': None,
                    'depth_img': None,
                    'ir_img': None
                },
                'InOutDoorPeopleRGBD': {
                    'img': None,
                    'depth_img': None,
                    'ir_img': None
                }
            },
            mod_list=['img', 'ir_img', 'depth_img'],
            to_float32=False,
            color_type='color',
            file_client_args=dict(backend='disk'),
            is_replace_img=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.mod_path_mapping_dict = mod_path_mapping_dict
        self.mod_list = mod_list
        self.is_replace_img = is_replace_img

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # assert self.file_client_args['backend'] == 'petrel', 'if you are not using ceph, please use the pipeline of LoadImageFromFile to get images. Otherwise, set the backend as petrel'
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results.get('ceph_prefix', None) is not None:
            filename = osp.join(results['ceph_prefix'],
                                results['img_info']['filename'])
        elif results['img_prefix'] is not None and len(
                results['img_prefix']) > 0:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_fields_list = []
        valid_image_list = []

        for dataset_key in self.mod_path_mapping_dict.keys():
            if dataset_key in filename:
                for mod_key in self.mod_path_mapping_dict[dataset_key].keys(
                ):
                    org_key = self.mod_path_mapping_dict[dataset_key][
                        mod_key]['org_key']
                    target_key = self.mod_path_mapping_dict[dataset_key][
                        mod_key]['target_key']

                    if org_key in filename:
                        curr_filename = filename.replace(org_key, target_key)
                        if mod_key == 'gray_optical_flow':
                            curr_filename = curr_filename.replace(
                                '/img1', '')
                            curr_filename = curr_filename.replace(
                                '.jpg', '_gray.jpg')
                            img_bytes = self.file_client.get(curr_filename)
                            img = mmcv.imfrombytes(img_bytes,
                                                   flag=self.color_type)
                        elif mod_key == 'optical_flow_x':
                            curr_filename = curr_filename.replace(
                                '/img1', '')
                            curr_filename = curr_filename.replace(
                                '.jpg', '.npy')
                            img_bytes = self.file_client.get(curr_filename)
                            with BytesIO(img_bytes) as buffer:
                                flow_data = np.load(buffer)
                            img = np.expand_dims(flow_data[..., 0],
                                                 -1).repeat(3, axis=-1)
                        elif mod_key == 'optical_flow_y':
                            curr_filename = curr_filename.replace(
                                '/img1', '')
                            curr_filename = curr_filename.replace(
                                '.jpg', '.npy')
                            img_bytes = self.file_client.get(curr_filename)
                            with BytesIO(img_bytes) as buffer:
                                flow_data = np.load(buffer)
                            img = np.expand_dims(flow_data[..., 1],
                                                 -1).repeat(3, axis=-1)
                        elif mod_key == 'waymo_lidar_img' or mod_key == 'waymo_lidar':
                            curr_filename = curr_filename.replace(
                                '.jpg', '.npy')
                            img_bytes = self.file_client.get(curr_filename)
                            with BytesIO(img_bytes) as buffer:
                                flow_data = np.load(buffer)
                            img = np.expand_dims(flow_data,
                                                 -1).repeat(3, axis=-1)
                        elif mod_key == 'depth_img':
                            img_bytes = self.file_client.get(curr_filename)
                            img = mmcv.imfrombytes(img_bytes,
                                                   flag='unchanged')
                            img = np.expand_dims(img, -1).repeat(3, axis=-1)
                        elif mod_key == 'waymo_lidar_map':
                            img_bytes = self.file_client.get(curr_filename)
                            img = mmcv.imfrombytes(img_bytes,
                                                   flag='unchanged')
                            img = np.expand_dims(img, -1).repeat(3, axis=-1)
                            # print('img shape:',img.shape,flush=True)
                        else:
                            img_bytes = self.file_client.get(curr_filename)
                            img = mmcv.imfrombytes(img_bytes,
                                                   flag=self.color_type)
                        if self.is_replace_img and (
                                mod_key == 'waymo_lidar_img'
                                or mod_key == 'waymo_lidar_map'):
                            mod_key = 'img'
                        img_fields_list.append(mod_key)
                        valid_image_list.append(mod_key)
                        results[mod_key] = img

        if len(img_fields_list) == 0:
            print('-' * 60)
            print('Can not final valid image')
            print('-' * 60)
            sys.exit(0)

        for mod_key in self.mod_list:
            if mod_key not in img_fields_list:
                results[mod_key] = results[img_fields_list[0]] * 0
                img_fields_list.append(mod_key)
        if self.to_float32:
            for img_key in img_fields_list:
                results[img_key] = results[img_key].astype(np.float32)
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = img_fields_list
        results['valid_img_fields'] = valid_image_list
        return results
