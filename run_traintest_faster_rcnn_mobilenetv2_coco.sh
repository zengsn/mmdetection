#!/bin/bash

# Download the configs
# https://github.com/UESTC-HZ/Project_LF/tree/main/benchmark/coco 
# Save them to :
# configs/faster_rcnn/faster_rcnn_mobilenetv2_coco.py
# configs/mask_rcnn/mask_rcnn_mobilnetv2_coco.py 
# configs/centernet/centernet_mobilenetv2_dcnv2_140e_coco.py

MODEL_NAME="faster_rcnn"
CONF_FNAME="faster_rcnn_mobilenetv2_coco"

# Train the models one by one
python tools/train.py \
	configs/${MODEL_NAME}/${CONF_FNAME}.py \
	--work-dir train_${CONF_FNAME} \
	--auto-resume

# INPUT_SHAPE="1333 800"

# Test: Generate results in .pkl
RESULT_PKL=train_${CONF_FNAME}_results.pkl
if [ -f "$RESULT_PKL" ]; then
    echo "Results exist in ${RESULT_PKL}."
else 
    echo "Test to get results: "
    python tools/test.py \
	configs/${MODEL_NAME}/${CONF_FNAME}.py \
	train_${CONF_FNAME}/epoch_12.pth \
	--out ${RESULT_PKL} 
    echo "Results saved to ${RESULT_PKL}. "
fi


