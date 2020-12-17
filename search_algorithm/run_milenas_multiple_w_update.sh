#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_milenas_multiple_w_update.sh "0" 44445 0.2 50 "saved_models" 0.05 0.0003 11 0.45

GPU=$1
run_id=$2
LAMBDA_VALID=$3
EPOCH=$4
MODEL_FILE=$5
LR=$6
ARCH_LR=$7
GROUP_ID=$8
PORTION=$9

python3 ./search_algorithm/train_milenas_multiple_w_update.py \
--gpu $GPU \
--run_id $run_id \
--unrolled \
--optimization V2 \
--arch_search_method DARTS \
--lambda_train_regularizer 1 \
--lambda_valid_regularizer $LAMBDA_VALID \
--batch_size 64 \
--epochs $EPOCH \
--layers 8 \
--model_path $MODEL_FILE \
--learning_rate $LR \
--arch_learning_rate $ARCH_LR \
--early_stopping 0 \
--group_id $GROUP_ID \
--mixed_portion $PORTION