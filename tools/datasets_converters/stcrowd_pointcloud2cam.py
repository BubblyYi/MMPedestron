import copy
import json
import multiprocessing
import os
import threading
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('agg')


def save_json(list_file, path):
    with open(path, 'w') as f:
        json.dump(list_file,
                  f,
                  indent=2,
                  ensure_ascii=False)
    print('save json done:', path)
    return 0


def load_jsonfile(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    return json_file


def trans_coordinate_tools(
        all_points, projection_matrix, r0_rect_matrix, tr_velo_to_cam_matrix):

    points = all_points[:, 0:3]  # lidar xyz (front, left, up)
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    new_points = np.dot(
        np.dot(
            np.dot(
                projection_matrix,
                r0_rect_matrix),
            tr_velo_to_cam_matrix),
        velo)
    return new_points


def get_lidar_mask(lidar_points, out_name='', cmap='gray_r'):
    # print('all_points shape:', lidar_points.shape)
    threadLock.acquire()
    lidar_points = np.delete(lidar_points, np.where(
        lidar_points[2, :] < 0), axis=1)
    lidar_points[:2] /= lidar_points[2, :]
    plt.figure(
        figsize=(128, 72),
        dpi=10,
    )
    IMG_H, IMG_W = 720, 1280
    # restrict canvas in range
    plt.axis([0, IMG_W, IMG_H, 0])
    u, v, z = lidar_points
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    lidar_points = np.delete(lidar_points, np.where(outlier), axis=1)
    # generate color map from depth
    u, v, z = lidar_points

    normal_z = (z - z.min()) / (z.max() - z.min()) * 255
    plt.scatter(
        [u],
        [v],
        c=[normal_z],
        cmap=cmap,
        alpha=1,
        s=1000,
        vmin=0,
        vmax=255)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.savefig(out_name, facecolor='black')
    plt.clf()
    plt.cla()
    plt.close()
    print('save lidar mask to:', out_name)
    threadLock.release()


def gen_all_lidar_mask(
        ann_path='',
        camera_info_path='',
        data_root_prefix='/data/STCrowd/'):

    ann_json = load_jsonfile(ann_path)
    camera_info = load_jsonfile(camera_info_path)
    projection_matrix = np.array(camera_info['p'])
    r0_rect_matrix = np.array(camera_info['r'])
    tr_velo_to_cam_matrix = np.array(camera_info['t'])
    new_image_list = []
    for img_info in ann_json['images']:
        # print('img_info:',img_info)
        new_img_item = copy.deepcopy(img_info)
        lidar_binary_path = data_root_prefix + img_info['point_cloud_path']
        img_path = img_info['file_name']
        curr_lidar_mask_out_prefix = data_root_prefix + \
            os.path.dirname(img_path.replace('left', 'pcd_mask'))
        if not os.path.exists(curr_lidar_mask_out_prefix):
            os.makedirs(curr_lidar_mask_out_prefix)
        curr_lidar_mask_out_path = os.path.join(
            curr_lidar_mask_out_prefix, os.path.basename(img_path))
        new_img_item['lidar_mask_name'] = img_path.replace('left', 'pcd_mask')
        new_image_list.append(new_img_item)
        lidar_points = np.fromfile(
            lidar_binary_path, dtype=np.float32).reshape(
            (-1, 4))
        trans_lidar_points = trans_coordinate_tools(lidar_points,
                                                    projection_matrix,
                                                    r0_rect_matrix,
                                                    tr_velo_to_cam_matrix)
        get_lidar_mask(trans_lidar_points, out_name=curr_lidar_mask_out_path)
    ann_json['images'] = new_image_list
    out_ann_path = os.path.join(
        os.path.dirname(ann_path),
        os.path.basename(ann_path).replace(
            '.json',
            '_lidar_mask.json'))
    save_json(ann_json, out_ann_path)


threadLock = threading.RLock()


def cal_lidar_map_job(img_info):
    lidar_binary_path = data_root_prefix + img_info['point_cloud_path']
    img_path = img_info['file_name']
    curr_lidar_mask_out_prefix = data_root_prefix + \
        os.path.dirname(img_path.replace('left', 'pcd_mask_test'))
    curr_lidar_mask_out_path = os.path.join(
        curr_lidar_mask_out_prefix,
        os.path.basename(img_path))
    if not os.path.exists(curr_lidar_mask_out_prefix):
        os.makedirs(curr_lidar_mask_out_prefix)
    lidar_points = np.fromfile(
        lidar_binary_path, dtype=np.float32).reshape(
        (-1, 4))
    trans_lidar_points = trans_coordinate_tools(lidar_points,
                                                projection_matrix,
                                                r0_rect_matrix,
                                                tr_velo_to_cam_matrix)
    get_lidar_mask(trans_lidar_points, out_name=curr_lidar_mask_out_path)


ann_path = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/STCrowd_Raw/val_set.json'
ann_path2 = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/STCrowd_Raw/train_set.json'
camera_info_path = 'YOUR_DTAT_ROOT_PATH/mmpedestron_datasets_ann/STCrowd_Raw/camera.json'
data_root_prefix = 'YOUR_DTAT_ROOT_PATH/mmpedestron_images/STCrowd/'
ann_json = load_jsonfile(ann_path)
ann_json2 = load_jsonfile(ann_path2)
camera_info = load_jsonfile(camera_info_path)
projection_matrix = np.array(camera_info['p'])
r0_rect_matrix = np.array(camera_info['r'])
tr_velo_to_cam_matrix = np.array(camera_info['t'])
maximum_processes_num = 8

if maximum_processes_num >= multiprocessing.cpu_count():
    maximum_processes_num = multiprocessing.cpu_count() - 1

ids = range(len(ann_json['images']))
choice_size = 100
image_check_list = np.sort(
    np.random.choice(
        ids,
        choice_size,
        replace=False)).tolist()

print('org image size:', len(ids))
print('image_check_list size:', len(image_check_list))
print('image_check_list:', image_check_list[:10])

with Pool(maximum_processes_num) as p:
    p.map(cal_lidar_map_job, ann_json['images'])

with Pool(maximum_processes_num) as p:
    p.map(cal_lidar_map_job, ann_json2['images'])
