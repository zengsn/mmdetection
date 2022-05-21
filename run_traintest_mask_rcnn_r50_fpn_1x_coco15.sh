#!/bin/bash

# Download the configs
# https://github.com/UESTC-HZ/Project_LF/tree/main/benchmark/coco 
# Save them to :
# configs/faster_rcnn/faster_rcnn_mobilenetv2_coco.py
# configs/mask_rcnn/mask_rcnn_mobilnetv2_coco.py 
# configs/centernet/centernet_mobilenetv2_dcnv2_140e_coco.py

INPUT_SHAPE="1333 800"
MODEL_NAME="mask_rcnn"
CONF_FNAME="mask_rcnn_r50_fpn_1x_coco15"

# Train the models one by one
WORK_DIR=train_${CONF_FNAME}
PTH_FILE=${WORK_DIR}/epoch_12.pth
if [ -f "$PTH_FILE" ]; then
    echo "Train result exist in ${PTH_FILE}, skip training."
else
    python tools/train.py \
	configs/${MODEL_NAME}/${CONF_FNAME}.py \
	--work-dir ${WORK_DIR} \
	--auto-resume
fi 

# Get flops
echo "Get model complexity in flops: "
python tools/analysis_tools/get_flops.py \
	configs/${MODEL_NAME}/${CONF_FNAME}.py \
	--shape ${INPUT_SHAPE} > ${WORK_DIR}_flops.txt
cat ${WORK_DIR}_flops.txt

# Test: Generate results in .pkl
RESULT_PKL=${WORK_DIR}_results.pkl
if [ -f "$RESULT_PKL" ]; then
    echo "Results exist in ${RESULT_PKL}."
else 
    echo "Test to get results: "
    python tools/test.py \
	configs/${MODEL_NAME}/${CONF_FNAME}.py \
	${PTH_FILE} \
	--out ${RESULT_PKL} 
    echo "Results saved to ${RESULT_PKL}. "
fi

# Result analysis
echo "Analyze the results: "
python tools/analysis_tools/analyze_results.py \
	configs/${MODEL_NAME}/${CONF_FNAME}.py \
	${RESULT_PKL} \
	${WORK_DIR}_results_figs \
	--topk 50


