#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 sh search_algorithm/run_gdas.sh "0" 0.025 0.001 0.0003 8


GPU=$1
LR_MAX=$2
LR_MIN=$3
LR_ARC=$4
LAYERS=$5

python3 ./search_algorithm/train_gdas.py \
--gpu $GPU \
--unrolled \
--optimization AOS \
--arch_search_method GDAS \
--batch_size 128 \
--tau_max 10 --tau_min 1 \
--epochs 250 \
--learning_rate $LR_MAX \
--learning_rate_min $LR_MIN \
--momentum 0.9 \
--weight_decay 0.0003 \
--arch_learning_rate $LR_ARC \
--layers $LAYERS