"""Get test image metas on a specific dataset.

Here is an example to run this script.

Example:
    python tools/misc/get_image_metas.py ${CONFIG} \
    --out ${OUTPUT FILE NAME}
"""
import argparse
import csv
import os
import os.path
import os.path as osp
from multiprocessing import Pool
import json

import mmcv
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Extract COCO data by a subset of classes.')
    parser.add_argument(
        '--data-dir',
        default='/data/coco-subset-15',
        help='The dir to save the subset dataset.')
    parser.add_argument(
        '--data-coco',
        default='/data/coco-dataset'
    )
    args = parser.parse_args()
    return args

def extract_labels_by_classes(source_labels, classes):
    print(source_labels["annotations"][0])
    print(source_labels["images"][0])
    print(source_labels["licenses"][0])

    subset_labels = {}  # output
    # {
    # "info": info, "images": [image], "annotations": [annotation], "licenses": [license], "categories": [category]
    # }
    subset_labels["info"] = source_labels["info"]
    # license{
    #    "url":"http://creativecommons.org/licenses/by-nc-sa/2.0/",
    #    "id":1,
    #    "name":"Attribution-NonCommercial-ShareAlike License"
    # }
    subset_labels["licenses"] = source_labels["licenses"]
    subset_labels["annotations"] = []
    subset_labels["images"] = []
    subset_labels["categories"] = []
    # print(source_labels["categories"])
    # categories[{
    # "id": int, "name": str, "supercategory": str,
    # }]
    class_ids = []
    for category in source_labels["categories"]:
        if category["name"] in classes:
            class_ids.append(category["id"])
            # Re-map the category id
            category["id"] = classes.index(category["name"]) + 1
            subset_labels["categories"].append(category)
    print("Use the categories: ")
    print(subset_labels["categories"])

    # annotation{
    # "id": int, "image_id": int, "category_id": int,
    # "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
    # }
    # image{
    #    "license":3,
    #    "file_name":"000000391895.jpg",
    #    "coco_url":"http://images.cocodataset.org/train2017/000000391895.jpg",
    #    "height":360,
    #    "width":640,
    #    "date_captured":"2013-11-14 11:18:45",
    #    "flickr_url":"http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
    #    "id":391895
    # }
    # print(len(source_labels["annotations"]))
    # print(len(source_labels["images"]))
    print("Process the annotations: ")
    image_ids = []
    for idx, annotation in enumerate(source_labels["annotations"]):
        print("-> index=%d \tin cat=%d \t" % (idx, annotation["category_id"]), end='')
        if annotation["category_id"] in class_ids:
            print("... accepted. \tvvv \t", end='')
            new_category_id = class_ids.index(annotation["category_id"]) + 1
            annotation["category_id"] = new_category_id  # Re-map the cat id
            subset_labels["annotations"].append(annotation)
            # subset_labels["images"].append(source_labels["images"][idx])
            if annotation["image_id"] not in image_ids:  # n-labels -> 1 images
                print("... new image id=%d " % (annotation["image_id"]))
                image_ids.append(annotation["image_id"])
            else:
                print("... repeated image.")
        else:
            print("... ignored. \txxx ")

    # Images
    total_num = len(image_ids)
    print("Process the chosen %d images: " % total_num, end='')
    for idx, img in enumerate(source_labels["images"]):
        if img["id"] in image_ids:
            subset_labels["images"].append(img)
            if len(subset_labels["images"]) % 5000 == 0:
                print("...%d" % idx, end='')
    print("end.")

    return subset_labels


def save_to_json(json_path, json_object):
    with open(json_path, "w") as f:
        json.dump(json_object, f)


def copy_images(coco15_labels, coco_dir, coco15_dir):
    for img in coco15_labels["images"]:
        img1_path = os.path.join(coco_dir, img["file_name"])
        img2_path = os.path.join(coco15_dir, img["file_name"])
        copy_cmd = "cp " + img1_path + " " + img2_path
        os.system(copy_cmd)
        print(copy_cmd)


def main():
    args = parse_args()
    print(args)
    # TODO: support passing by args
    classes = ('person', 'bicycle', 'motorcycle', 'fire hydrant',
               'bench', 'cat', 'dog',
               'backpack', 'umbrella', 'frisbee',
               'kite', 'skateboard', 'chair',
               'potted plant', 'sink')
    class_ids = (1, 2, 4, 11, 14, 16, 17, 25, 26, 30, 34, 37, 57, 59, 72)

    # List files
    # data_root_dir = args.data_root  # /data
    data_coco_dir = args.data_coco  # /data/coco-dataset
    data_coco_anno_dir = os.path.join(data_coco_dir, 'annotations')
    data_coco_train_dir = os.path.join(data_coco_dir, 'train2017')
    data_coco_test_dir = os.path.join(data_coco_dir, 'test2017')
    data_coco_val_dir = os.path.join(data_coco_dir, 'val2017')
    data_coco15_dir = args.data_dir  # output dir
    if not os.path.exists(data_coco15_dir):
        os.makedirs(data_coco15_dir)
    data_coco15_anno_dir = os.path.join(data_coco15_dir, 'annotations')
    if not os.path.exists(data_coco15_anno_dir):
        os.makedirs(data_coco15_anno_dir)

    # Load instances labels
    # - train
    instances_train_sub_labels_file = os.path.join(data_coco15_anno_dir, 'instances_train2017.json')
    if not os.path.exists(instances_train_sub_labels_file):
        instances_train_labels_file = open(os.path.join(data_coco_anno_dir, 'instances_train2017.json'))
        instances_train_labels = json.load(instances_train_labels_file)  # instances train label
        instances_train_sub_labels = extract_labels_by_classes(instances_train_labels, classes)
        save_to_json(instances_train_sub_labels_file, instances_train_sub_labels)
    else:
        print("!!! Already extracted at: %s" % instances_train_sub_labels_file)
        instances_train_sub_labels = json.load(instances_train_sub_labels_file)
    # - val
    instances_val_sub_labels_file = os.path.join(data_coco15_anno_dir, 'instances_val2017.json')
    if not os.path.exists(instances_val_sub_labels_file):
        instances_val_labels_file = open(os.path.join(data_coco_anno_dir, 'instances_val2017.json'))
        instances_val_labels = json.load(instances_val_labels_file)  # instances val label
        instances_val_sub_labels = extract_labels_by_classes(instances_val_labels, classes)
        save_to_json(instances_val_sub_labels_file, instances_val_sub_labels)
    else:
        print("!!! Already extracted at: %s" % instances_train_sub_labels_file)
        instances_val_sub_labels = json.load(instances_val_sub_labels_file)

    # Load keypoints labels
    # - train
    keypoints_train_sub_labels_file = os.path.join(data_coco15_anno_dir, 'person_keypoints_train2017.json')
    if not os.path.exists(keypoints_train_sub_labels_file):
        keypoints_train_labels_file = open(os.path.join(data_coco_anno_dir, 'person_keypoints_train2017.json'))
        keypoints_train_labels = json.load(keypoints_train_labels_file)  # keypoints train label
        keypoints_train_sub_labels = extract_labels_by_classes(keypoints_train_labels, classes)
        save_to_json(keypoints_train_sub_labels_file, keypoints_train_sub_labels)
    else:
        print("!!! Already extracted at: %s" % keypoints_train_sub_labels_file)
        # keypoints_train_sub_labels = json.load(keypoints_train_sub_labels_file)
    # - val
    keypoints_val_sub_labels_file = os.path.join(data_coco15_anno_dir, 'person_keypoints_val2017.json')
    if not os.path.exists(keypoints_val_sub_labels_file):
        keypoints_val_labels_file = open(os.path.join(data_coco_anno_dir, 'person_keypoints_val2017.json'))
        keypoints_val_labels = json.load(keypoints_val_labels_file)  # keypoints val label
        keypoints_val_sub_labels = extract_labels_by_classes(keypoints_val_labels, classes)
        save_to_json(keypoints_val_sub_labels_file, keypoints_val_sub_labels)
    else:
        print("!!! Already extracted at: %s" % keypoints_val_sub_labels_file)
        # keypoints_val_sub_labels = json.load(keypoints_val_sub_labels_file)

    # Load captions labels
    # - train
    # captions_train_labels_file = open(os.path.join(data_coco_anno_dir, 'captions_train2017.json'))
    # captions_train_labels = json.load(captions_train_labels_file)  # captions train label
    # captions_train_sub_labels = extract_labels_by_classes(captions_train_labels, classes)
    # save_to_json(os.path.join(data_coco15_anno_dir, 'captions_train2017.json'), captions_train_sub_labels)
    # - val
    # captions_val_labels_file = open(os.path.join(data_coco_anno_dir, 'captions_val2017.json'))
    # captions_val_labels = json.load(captions_val_labels_file)  # captions val label
    # captions_val_sub_labels = extract_labels_by_classes(captions_val_labels, classes)
    # save_to_json(os.path.join(data_coco15_anno_dir, 'captions_val2017.json'), captions_val_sub_labels)

    # Copy images
    # - train
    data_coco15_train_dir = os.path.join(data_coco15_dir, 'train2017')
    if not os.path.exists(data_coco15_train_dir):
        os.makedirs(data_coco15_train_dir)
    copy_images(instances_train_sub_labels, data_coco_train_dir, data_coco15_train_dir)

    # - val
    data_coco15_val_dir = os.path.join(data_coco15_dir, 'val2017')
    if not os.path.exists(data_coco15_val_dir):
        os.makedirs(data_coco15_val_dir)
    copy_images(instances_val_sub_labels, data_coco_val_dir, data_coco15_val_dir)

    # - test NO NEED
    # data_coco15_test_dir = os.path.join(data_coco15_dir, 'test2017')
    # if not os.path.exists(data_coco15_test_dir):
    #    os.makedirs(data_coco15_test_dir)


if __name__ == '__main__':
    main()