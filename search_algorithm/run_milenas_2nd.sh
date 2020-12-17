#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_milenas_2nd.sh "2" 307 1 1 50 8 "saved_models"


GPU=$1
run_id=$2
LAMBDA_TRAIN=$3
LAMBDA_VALID=$4
EPOCH=$5
LAYERS=$6
MODEL_FILE=$7

python3 ./search_algorithm/train_milenas_2nd.py \
--gpu $GPU \
--run_id $run_id \
--unrolled \
--optimization V2_2ndOrder \
--arch_search_method DARTS \
--lambda_train_regularizer $LAMBDA_TRAIN \
--lambda_valid_regularizer $LAMBDA_VALID \
--batch_size 64 \
--epochs $EPOCH \
--layers $LAYERS \
--model_path $MODEL_FILE