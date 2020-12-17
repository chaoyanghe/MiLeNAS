#!/usr/bin/env bash
#-----------
#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_darts.sh "0" 77700 1 1 50 8 "saved_models" 1 0
#CUDA_VISIBLE_DEVICES=1 sh search_algorithm/run_darts.sh "1" 77701 1 1 50 8 "saved_models" 1 0
#CUDA_VISIBLE_DEVICES=2 sh search_algorithm/run_darts.sh "2" 77702 1 1 50 8 "saved_models" 1 0
#CUDA_VISIBLE_DEVICES=3 sh search_algorithm/run_darts.sh "3" 77703 1 1 50 8 "saved_models" 1 0

#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_darts.sh "0" 77710 1 1 50 8 "saved_models" 1 1
#CUDA_VISIBLE_DEVICES=1 sh search_algorithm/run_darts.sh "1" 77711 1 1 50 8 "saved_models" 1 1
#CUDA_VISIBLE_DEVICES=2 sh search_algorithm/run_darts.sh "2" 77712 1 1 50 8 "saved_models" 1 1
#CUDA_VISIBLE_DEVICES=3 sh search_algorithm/run_darts.sh "3" 77713 1 1 50 8 "saved_models" 1 1

#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_darts.sh "0" 77714 1 1 50 8 "saved_models" 1 1
#CUDA_VISIBLE_DEVICES=1 sh search_algorithm/run_darts.sh "1" 77715 1 1 50 8 "saved_models" 1 1

GPU=$1
run_id=$2
LAMBDA_TRAIN=$3
LAMBDA_VALID=$4
EPOCH=$5
LAYERS=$6
MODEL_FILE=$7
UNROLLED=$8
GROUP_ID=$9

python3 ./search_algorithm/train_darts.py \
--gpu $GPU \
--run_id $run_id \
--unrolled $UNROLLED \
--optimization DARTS \
--arch_search_method DARTS \
--lambda_train_regularizer $LAMBDA_TRAIN \
--lambda_valid_regularizer $LAMBDA_VALID \
--batch_size 64 \
--epochs $EPOCH \
--layers $LAYERS \
--model_path $MODEL_FILE \
--group_id $GROUP_ID