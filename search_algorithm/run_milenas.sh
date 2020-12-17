#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 ./search_algorithm/run_milenas.sh "0" 200003 1 50 "saved_models" 0.025 0.0003 2021 8

GPU=$1
run_id=$2
LAMBDA_VALID=$3
EPOCH=$4
MODEL_FILE=$5
LR=$6
ARCH_LR=$7
GROUP_ID=$8
# CHANNEL_SIZE=$9
LAYER_NUM=$9


python3 ./search_algorithm/train_milenas.py \
--gpu $GPU \
--run_id $run_id \
--unrolled \
--optimization V2 \
--arch_search_method DARTS \
--lambda_train_regularizer 1 \
--lambda_valid_regularizer $LAMBDA_VALID \
--batch_size 64 \
--epochs $EPOCH \
--layers $LAYER_NUM \
--model_path $MODEL_FILE \
--learning_rate $LR \
--arch_learning_rate $ARCH_LR \
--early_stopping 0 \
--group_id $GROUP_ID
# --init_channels $CHANNEL_SIZE