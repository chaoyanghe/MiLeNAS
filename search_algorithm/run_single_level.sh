#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_single_level.sh "0" 77723 1 50 8 "saved_models" 0.025 2


GPU=$1
run_id=$2
LAMBDA_VALID=$3
EPOCH=$4
LAYERS=$5
MODEL_FILE=$6
LR=$7
GROUP_ID=$8

python3 ./search_algorithm/train_single_level.py \
--gpu $GPU \
--run_id $run_id \
--unrolled \
--optimization SINGLE \
--arch_search_method DARTS \
--lambda_valid_regularizer $LAMBDA_VALID \
--batch_size 64 \
--epochs $EPOCH \
--layers $LAYERS \
--model_path $MODEL_FILE \
--learning_rate $LR \
--group_id $GROUP_ID