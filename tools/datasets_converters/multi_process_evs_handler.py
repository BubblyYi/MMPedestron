# @Time    : 05/06/2023 14:21
# @Author  : BubblyYi
# @FileName: multi_process_evs_handler.py
# @Software: PyCharm

import argparse
import json
import multiprocessing
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
import pandas as pd
import PIL.Image as Image
from pycocotools.coco import COCO


def make_folder(folder_path):
    """make dir.

    :param folder_path: dir path
    :return: none
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print('make dir: %s done' % folder_path)


def save_json(list_file, path):
    with open(path, 'w') as f:
        json.dump(list_file,
                  f,
                  indent=1,
                  ensure_ascii=False)
    print('save json done:', path)
    return 0


def load_jsonfile(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    return json_file


def dfs_find_files(rootdir, ext_list=['.dat']):
    pth_fn = []
    org_root_dir = rootdir + ''

    def findfile(rootdir):
        for fileordir in os.scandir(rootdir):
            if fileordir.is_dir():
                findfile(fileordir.path)
            else:
                ext = os.path.splitext(fileordir)[1]
                if ext in ext_list:
                    relpath = os.path.relpath(fileordir, org_root_dir)
                    pth_fn.append(relpath)

    findfile(rootdir)
    return pth_fn


def save_json_ann(final_img_list, final_ann_list, out_name=''):
    instance = {}
    images = []
    annotations = []
    categories = [
        {'id': 1, 'name': 'person'}
    ]

    new_ann_id = 0
    new_img_id = 0
    for image, ann_bboxes in zip(final_img_list, final_ann_list):
        img_item = {}
        img_item['file_name'] = image
        img_item['width'] = 346
        img_item['height'] = 260
        img_item['id'] = new_img_id
        images.append(img_item)

        for ann_bbox in ann_bboxes:
            ann = {}
            ann['image_id'] = new_img_id
            ann['category_id'] = 1
            ann['bbox'] = ann_bbox
            ann['id'] = new_ann_id
            ann['area'] = int(ann_bbox[2] * ann_bbox[3])
            ann['ignore'] = 0
            ann['iscrowd'] = 0
            new_ann_id += 1
            annotations.append(ann)
        new_img_id += 1
    instance['images'] = images
    instance['annotations'] = annotations
    instance['categories'] = categories

    save_json(instance, out_name)
    print('double check')
    new_coco = COCO(out_name)
    print('new_coco image list:', len(list(new_coco.imgs.keys())))


def evs_handler(dataset_numpy_item):
    print('dataset_numpy_item:', dataset_numpy_item)

    # Defining the camera resolution
    img_w = 346
    img_h = 260

    # Defining the time interval to build the SAE
    time_interval = 40e3

    dataset_numpy_path = os.path.join(img_prefix, dataset_numpy_item)
    dataset_xml_path = dataset_numpy_path.replace('.npy', '.xml')
    dataset_xml_path = dataset_xml_path.replace('/numpy', '/xml')

    # BBOX
    # parse xml file
    tree = ET.parse(dataset_xml_path)
    root = tree.getroot()  # get root object
    bbox_list = []
    img_list = []
    for member in root.findall('object'):
        class_name = member[0].text  # class name
        if class_name == 'person':
            # bbox coordinates
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            # store data in list
            bbox_list.append(
                [xmin, ymin, max(xmax - xmin, 0),
                 max(ymax - ymin, 0)])
    # SAE
    events = np.load(dataset_numpy_path)
    df_events = pd.DataFrame(
        {'timestamp': events[:, 0],
         'x': events[:, 1],
         'y': events[:, 2],
         'polarity': events[:, 3]})
    timestamps_vector = df_events['timestamp'].to_numpy()

    # Separating positive and negative events
    df_events_neg = df_events[df_events['polarity'] == 0]
    df_events_pos = df_events[df_events['polarity'] == 1]

    # Keeping only the last events per (x,y) - good for sae only
    df_events_neg_remaining = df_events_neg.sort_values(
        by='timestamp').drop_duplicates(
        subset=['x', 'y'],
        keep='last', inplace=False)
    df_events_pos_remaining = df_events_pos.sort_values(
        by='timestamp').drop_duplicates(
        subset=['x', 'y'],
        keep='last', inplace=False)

    # Creating an empty time surface with two channels (positive and negative)
    sae = np.zeros((img_w, img_h, 2), dtype='float32')
    # Selecting time_limit as the timestamp of the last event in the sample
    time_limit = int(timestamps_vector[-1])
    # Selecting t_init_0 as the difference between time_limit and the time
    # interval (in this case is 40 ms)
    t_init_0 = int(timestamps_vector[-1] - time_interval)

    # Considering only the last event occurred in each pixel - This approach
    # is suitable only for SAE
    df_events_neg_remaining_subset = df_events_neg_remaining[
        df_events_neg_remaining['timestamp'].isin(range(t_init_0, time_limit))]
    df_events_pos_remaining_subset = df_events_pos_remaining[
        df_events_pos_remaining['timestamp'].isin(range(t_init_0, time_limit))]

    # Filling the negative surface - SAE
    x_neg = df_events_neg_remaining_subset['x'].to_numpy()
    y_neg = df_events_neg_remaining_subset['y'].to_numpy()
    t_neg = df_events_neg_remaining_subset['timestamp'].to_numpy()
    sae[x_neg, y_neg, 1] = (
        255 * ((t_neg - t_init_0) / time_interval)).astype(int)

    # Filling the positive surface - SAE
    x_pos = df_events_pos_remaining_subset['x'].to_numpy()
    y_pos = df_events_pos_remaining_subset['y'].to_numpy()
    t_pos = df_events_pos_remaining_subset['timestamp'].to_numpy()
    sae[x_pos, y_pos, 0] = (
        255 * ((t_pos - t_init_0) / time_interval)).astype(int)
    im = Image.fromarray(
        0.5 * sae[:, :, 0].T + 0.5 * sae[:, :, 1].T).convert('L')

    frame_name = os.path.basename(dataset_numpy_path).replace('.npy', '.png')
    img_name = frame_name
    im.save(out_img_prefix + img_name)
    img_list.append(img_name)

    res_dict = {}
    res_dict['img_list'] = img_list
    res_dict['bbox_list'] = [bbox_list]
    return res_dict


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('--img_prefix', default='numpy/val/', type=str)
    parser.add_argument('--out_img_prefix', default='val_img/', type=str)
    parser.add_argument('--out_json_name', default='pedro_val.json', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    dataset_prefix = 'YOUR_DTAT_ROOT_PATH/mmpedestron_images/PEDRo_events_dataset/'

    ann_out_prefix = dataset_prefix + 'coco_ann/'
    img_prefix = dataset_prefix + ARGS.img_prefix
    out_img_prefix = dataset_prefix + ARGS.out_img_prefix

    make_folder(ann_out_prefix)
    make_folder(out_img_prefix)

    file_path_list = dfs_find_files(img_prefix, ext_list=['.npy'])

    print('file_path_list:', file_path_list)

    maximum_processes_num = 10

    if maximum_processes_num >= multiprocessing.cpu_count():
        maximum_processes_num = multiprocessing.cpu_count() - 1
    with Pool(maximum_processes_num) as p:
        out_res = p.map(evs_handler, file_path_list)

    final_img_list = []
    final_ann_list = []

    for res in out_res:
        final_img_list.extend(res['img_list'])
        final_ann_list.extend(res['bbox_list'])
    print('len final_img_list:', len(final_img_list))
    print('len final_ann_list:', len(final_ann_list))

    save_json_ann(final_img_list,
                  final_ann_list,
                  out_name=ann_out_prefix + ARGS.out_json_name)
