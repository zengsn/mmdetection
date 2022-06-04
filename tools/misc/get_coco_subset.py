"""Get a subset from the COCO dataset by specifying the chosen classes.

Here is an example to run this script.

Example:
    python get_coco_subset.py \
    --data-dir /dir/to/save/the/subset/of/coco/
    --data-coco /dir/of/the/original/coco/
    --max-bbox 5
    [--stats-only True]
"""
import argparse
import os
import os.path
import json
import pprint

DEF_MAX_BBOX = 0  # no limit
DEF_MAX_SAMPLES = 5000


def parse_args():
    parser = argparse.ArgumentParser(description='Extract COCO data by a subset of classes.')
    parser.add_argument(
        '--data-dir',
        default='/data/coco15',
        help='The dir to save the subset dataset.')
    parser.add_argument(
        '--data-coco',
        default='/data/coco-dataset',
        help='The dir of the original COCO dataset.')
    parser.add_argument(
        '--max-bbox',
        default=DEF_MAX_BBOX,
        type=int,
        help='The maximal annotations on one image.')
    parser.add_argument(
        '--max-samples',
        default=DEF_MAX_SAMPLES,
        type=int,
        help='The maximal annotations on one image. (No supported yet!!!)')
    parser.add_argument(
        '--stats-only',
        type=bool, default=None,
        help='Stats the data only.')
    args = parser.parse_args()
    return args


def extract_labels_by_classes(source_labels, classes, max_bbox=DEF_MAX_BBOX):
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
    image_annotations = {}
    ignored_image_ids = []
    for idx, annotation in enumerate(source_labels["annotations"]):
        print("-> index=%d \tin cat=%d \t" % (idx, annotation["category_id"]), end='')
        if annotation["category_id"] in class_ids:
            this_image_id = annotation["image_id"]
            if this_image_id not in ignored_image_ids:
                # Check and accept images with no more than max_bbox annotations
                if (this_image_id in image_ids) \
                        and DEF_MAX_BBOX < max_bbox <= image_annotations[this_image_id]:
                    # Ignore this image
                    ignored_image_ids.append(this_image_id)
                    print("... ignored. \txxx \t")
                else:  # Accept this annotation
                    print("... accepted. \tvvv \t", end='')
                    new_category_id = class_ids.index(annotation["category_id"]) + 1
                    annotation["category_id"] = new_category_id  # Re-map the cat id
                    subset_labels["annotations"].append(annotation)
                    # subset_labels["images"].append(source_labels["images"][idx])
                    if annotation["image_id"] not in image_ids:  # n-labels -> 1 images
                        print("... new image id=%d " % (annotation["image_id"]))
                        image_ids.append(annotation["image_id"])
                        image_annotations[this_image_id] = 1
                    else:
                        print("... repeated image.")
                        image_annotations[this_image_id] += 1
            else:
                print("... ignored. \txxx ")
        else:
            print("... ignored. \txxx ")

    # Remove the ignored images and its annotations
    image_ids_left = []
    for image_id in image_ids:
        if image_id not in ignored_image_ids:
            image_ids_left.append(image_id)
        else:
            print("Image id=%d is removed." % image_id)
    image_ids = image_ids_left  # ignored image ids removed.
    subset_annotations = []
    for annotation in subset_labels["annotations"]:
        if annotation["image_id"] in image_ids:
            subset_annotations.append(annotation)
        else:
            print("Annotation %d of image %d is removed."
                  % (annotation["id"], annotation["image_id"]))
    subset_labels["annotations"] = subset_annotations

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


def stats(instances_train_labels):
    """Stats and show the subset of COCO"""
    # {
    # "info": info, "images": [image], "annotations": [annotation], "licenses": [license], "categories": [category]
    # }
    total_classes_num = len(instances_train_labels["categories"])
    print("A subset of COCO containing %d classes (Train)" % total_classes_num)
    samples_in_cat = {}
    labels_in_cat = {}
    image_ids = []
    for annotation in instances_train_labels["annotations"]:
        if annotation["category_id"] not in samples_in_cat.keys():
            samples_in_cat[annotation["category_id"]] = 0
        if annotation["category_id"] not in labels_in_cat.keys():
            labels_in_cat[annotation["category_id"]] = 0
        labels_in_cat[annotation["category_id"]] += 1  # Increase labels
        if annotation["image_id"] not in image_ids:
            image_ids.append(annotation["image_id"])
            samples_in_cat[annotation["category_id"]] += 1
    pprint.pprint(samples_in_cat)
    print(len(image_ids))
    # Print the stats
    stats_info = []
    print("ID \t Class \t\t Samples \t Labels (bbox)")
    for category in instances_train_labels["categories"]:
        print("%d " % category["id"], end='')
        if len(category["name"]) < 6:
            print("\t %s \t" % category["name"], end='')
        else:
            print("\t %s " % category["name"], end='')
        print("\t %d \t\t %d"
              % (samples_in_cat[category["id"]], labels_in_cat[category["id"]]))
        stats_info.append({
            "id": category["id"],
            "name": category["name"],
            "samples": samples_in_cat[category["id"]],
            "annotations": labels_in_cat[category["id"]]
        })
    return stats_info


def main():
    args = parse_args()
    print(args)
    # TODO: support passing by args
    classes = ('person', 'bicycle', 'motorcycle', 'fire hydrant',
               'bench', 'cat', 'dog',
               'backpack', 'umbrella', 'frisbee',
               'kite', 'skateboard', 'chair',
               'potted plant', 'sink')
    # class_ids = (1, 2, 4, 11, 14, 16, 17, 25, 26, 30, 34, 37, 57, 59, 72)

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
    instances_train_sub_labels_fpath = os.path.join(data_coco15_anno_dir, 'instances_train2017.json')
    if not os.path.exists(instances_train_sub_labels_fpath):
        instances_train_labels_file = open(os.path.join(data_coco_anno_dir, 'instances_train2017.json'))
        instances_train_labels = json.load(instances_train_labels_file)  # instances train label
        instances_train_sub_labels = extract_labels_by_classes(instances_train_labels, classes, args.max_bbox)
        save_to_json(instances_train_sub_labels_fpath, instances_train_sub_labels)
    else:
        print("!!! Already extracted at: %s" % instances_train_sub_labels_fpath)
        json_file = open(instances_train_sub_labels_fpath)
        instances_train_sub_labels = json.load(json_file)
        json_file.close()
    # - val
    instances_val_sub_labels_fpath = os.path.join(data_coco15_anno_dir, 'instances_val2017.json')
    if not os.path.exists(instances_val_sub_labels_fpath):
        instances_val_labels_file = open(os.path.join(data_coco_anno_dir, 'instances_val2017.json'))
        instances_val_labels = json.load(instances_val_labels_file)  # instances val label
        instances_val_sub_labels = extract_labels_by_classes(instances_val_labels, classes, args.max_bbox)
        save_to_json(instances_val_sub_labels_fpath, instances_val_sub_labels)
    else:
        print("!!! Already extracted at: %s" % instances_val_sub_labels_fpath)
        json_file = open(instances_val_sub_labels_fpath)
        instances_val_sub_labels = json.load(json_file)
        json_file.close()

    # Load keypoints labels
    # - train
    keypoints_train_sub_labels_fpath = os.path.join(data_coco15_anno_dir, 'person_keypoints_train2017.json')
    if not os.path.exists(keypoints_train_sub_labels_fpath):
        keypoints_train_labels_file = open(os.path.join(data_coco_anno_dir, 'person_keypoints_train2017.json'))
        keypoints_train_labels = json.load(keypoints_train_labels_file)  # keypoints train label
        keypoints_train_sub_labels = extract_labels_by_classes(keypoints_train_labels, classes, args.max_bbox)
        save_to_json(keypoints_train_sub_labels_fpath, keypoints_train_sub_labels)
    else:
        print("!!! Already extracted at: %s" % keypoints_train_sub_labels_fpath)
        # keypoints_train_sub_labels = json.load(keypoints_train_sub_labels_file)
    # - val
    keypoints_val_sub_labels_fpath = os.path.join(data_coco15_anno_dir, 'person_keypoints_val2017.json')
    if not os.path.exists(keypoints_val_sub_labels_fpath):
        keypoints_val_labels_file = open(os.path.join(data_coco_anno_dir, 'person_keypoints_val2017.json'))
        keypoints_val_labels = json.load(keypoints_val_labels_file)  # keypoints val label
        keypoints_val_sub_labels = extract_labels_by_classes(keypoints_val_labels, classes, args.max_bbox)
        save_to_json(keypoints_val_sub_labels_fpath, keypoints_val_sub_labels)
    else:
        print("!!! Already extracted at: %s" % keypoints_val_sub_labels_fpath)
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
    if args.stats_only is None:
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

    stats_info = stats(instances_train_sub_labels)
    save_to_json(os.path.join(data_coco15_dir, "stats_info.json"), stats_info)


if __name__ == '__main__':
    main()
