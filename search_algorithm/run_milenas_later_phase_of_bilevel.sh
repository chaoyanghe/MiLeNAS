#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_milenas_later_phase_of_bilevel.sh "0" 77763 1 50 8 "saved_models" 0.025 0.0003 6



GPU=$1
run_id=$2
LAMBDA_VALID=$3
EPOCH=$4
LAYERS=$5
MODEL_FILE=$6
LR=$7
ARCH_LR=$8
GROUP_ID=$9

python3 ./search_algorithm/train_milenas_later_phase_of_bilevel.py \
--gpu $GPU \
--run_id $run_id \
--unrolled \
--optimization V2 \
--arch_search_method DARTS \
--lambda_train_regularizer 1 \
--lambda_valid_regularizer $LAMBDA_VALID \
--batch_size 64 \
--epochs $EPOCH \
--layers $LAYERS \
--model_path $MODEL_FILE \
--learning_rate $LR \
--arch_learning_rate $ARCH_LR \
--early_stopping 0 \
--group_id $GROUP_ID