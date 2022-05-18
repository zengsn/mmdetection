# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_mobilenetv2_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15),
        mask_head=dict(num_classes=15)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('person', 'bicycle', 'motorcycle', 'fire hydrant',
               'bench', 'cat', 'dog',
               'backpack', 'umbrella', 'frisbee',
               'kite', 'skateboard', 'chair',
               'potted plant', 'sink')
data = dict(
    train=dict(
        img_prefix='lawn_in_coco/train2017/',
        classes=classes,
        ann_file='lawn_in_coco/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='lawn_in_coco/val2017/',
        classes=classes,
        ann_file='lawn_in_coco/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='lawn_in_coco/val2017/',
        classes=classes,
        ann_file='lawn_in_coco/annotations/instances_val2017.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'