#!/usr/bin/env bash
#----HKUST----
# 1.7M
#CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" v2_r2_e200 600 0.020
#CUDA_VISIBLE_DEVICES=8 sh run_eval_cnn.sh "8" v2_r2_e200 600 0.025
#CUDA_VISIBLE_DEVICES=9 sh run_eval_cnn.sh "9" v2_r2_e200 600 0.030
#CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" v2_r2_e200 600 0.050
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" v2_r2_e200 800 0.050
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" v2_r2_e200 800 0.040

#CUDA_VISIBLE_DEVICES=8 sh run_eval_cnn.sh "8" v2_r3_e200 600 0.020
#CUDA_VISIBLE_DEVICES=9 sh run_eval_cnn.sh "9" v2_r3_e200 600 0.025
#CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" v2_r3_e200 600 0.030
#CUDA_VISIBLE_DEVICES=6 sh run_eval_cnn.sh "6" v2_r3_e200 600 0.050

#----TODO-----
# 1.3M
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r4_e200 600 0.020
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r4_e200 600 0.025
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r4_e200 600 0.030
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r4_e200 600 0.050
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r4_e200 800 0.050

# 2.09M
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r5_e200 600 0.020
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r5_e200 600 0.025
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r5_e200 600 0.030
#CUDA_VISIBLE_DEVICES=x sh run_eval_cnn.sh "x" v2_r5_e200 600 0.050

# 1.72M
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" l1_r0_e200 600 0.020 (Running)
#CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" l1_r0_e200 600 0.025 (Running)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" l1_r0_e200 600 0.030 (Running)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" l1_r0_e200 600 0.050 (Running)

# 1.72M
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" l05_r2_e200 600 0.020 (Running)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" l05_r2_e200 600 0.025 (Running)
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" l05_r2_e200 600 0.030 (Running)
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" l05_r2_e200 600 0.050 (Running)

#----------------after bug fixed-------------

#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r100_e50_48_n7r1 600 0.030 (HN14)
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r101_e50_48_n7r3 600 0.030 (HN14)
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r102_e50_45_n8r1 600 0.030 (V100)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r103_e50_45_n7r2 600 0.030 (V100)

#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r116_e100_95_n3r0 600 0.030 (V100)
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r117_e100_98_n3r0 600 0.030 (V100)
#----------------finished-------------
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r118_e100_92_n4r1 600 0.030 (HN11)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r119_e100_86_n4r0 600 0.030 (HN11)

#----------------
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" v2_r2_e200 600 0.030 wandb/run-20190815_101758-ucn8bqf3/weights.pt

#evaluate huawei
# CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" r112_e50_48_n6_r2 600 0.030 saved_models

#evaluate e50 top2
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r101_e50_48_n7r3 600 0.030 saved_models (97.66)
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r101_e50_48_n7r3 600 0.030 saved_models (HN11)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r101_e50_48_n7r3 600 0.030 saved_models (HN13)
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r101_e50_48_n7r3 600 0.030 saved_models (HN13)
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r101_e50_48_n7r3 600 0.030 saved_models (HN13)
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r101_e50_48_n7r3 600 0.030 saved_models (HN14)
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r101_e50_48_n7r3 600 0.030 saved_models (HN14)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r101_e50_48_n7r3 600 0.030 saved_models (SJTU)
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r101_e50_48_n7r3 600 0.030 saved_models (SJTU)
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r101_e50_48_n7r3 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r101_e50_48_n7r3 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r101_e50_48_n7r3 600 0.030 saved_models

#evaluate e50 top1 (0.003)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r101_e50_48_n7r3 600 0.030 0.003 saved_models

#evaluate e100 top2
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r118_e100_92_n4r1 600 0.030 saved_models (Finished)
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=4 sh run_eval_cnn.sh "4" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=5 sh run_eval_cnn.sh "5" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=6 sh run_eval_cnn.sh "6" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=7 sh run_eval_cnn.sh "7" r118_e100_92_n4r1 600 0.030 saved_models (HKUST)
#CUDA_VISIBLE_DEVICES=8 sh run_eval_cnn.sh "8" r118_e100_92_n4r1 600 0.030 saved_models


#evaluate e50 top1 (0.003)
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r118_e100_92_n4r1 600 0.030 0.003 saved_models

#-----------------------------------------------
#evaluate e200 top2

# 4.4G
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r104_e200_185_n1r0_91_116 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_185_n1r0_91_116 600 0.030 saved_models


# 5.6G
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_197_n2r0_91_188 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_197_n2r0_91_188 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=5 sh run_eval_cnn.sh "5" r104_e200_197_n2r0_91_188 600 0.030 saved_models

#7.4G
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r104_e200_131_n3r0_90_528 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r104_e200_131_n3r0_90_528 600 0.030 saved_models

#8.5G
#CUDA_VISIBLE_DEVICES=4 sh run_eval_cnn.sh "4" r104_e200_120_n4r0_90_32 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r104_e200_120_n4r0_90_32 600 0.030 saved_models

#9.6G
#CUDA_VISIBLE_DEVICES=5 sh run_eval_cnn.sh "5" r104_e200_106_n5r0_89_776 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r104_e200_106_n5r0_89_776 600 0.030 saved_models

#11G
#CUDA_VISIBLE_DEVICES=6 sh run_eval_cnn.sh "6" r104_e200_50_n6r0_87_056 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=4 sh run_eval_cnn.sh "4" r104_e200_50_n6r0_87_056 600 0.030 saved_models

#>11G to evaluate
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r104_e200_43_n7r0_85_824 600 0.030 saved_models

#evaluation
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r306_2nd_e50_49_n7r8 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r306_2nd_e50_49_n7r8 600 0.030 saved_models

#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r803_2nd_e50_46_n7r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r803_2nd_e50_46_n7r1 600 0.025 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r803_2nd_e50_46_n7r1 600 0.050 saved_models

#-----------
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r904_e80_75_n4_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r902_e200_192_n2_r1 600 0.030 saved_models


#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_198_n2_r2 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r902_e200_155_n3_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r902_e200_136_n4_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r902_e200_104_n4_r2 600 0.030 saved_models

#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_62_n5_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r902_e200_64_n5_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r902_e200_42_n6_r0 600 0.030 saved_models

#-----------

#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r306_2nd_e50_49_n7r8 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r306_2nd_e50_49_n7r8 600 0.030 saved_models

# round 9
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_62_n5_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r902_e200_62_n5_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r902_e200_62_n5_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r902_e200_62_n5_r0 600 0.030 saved_models

#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_42_n6_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r902_e200_42_n6_r0 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r902_e200_42_n6_r0 600 0.030 saved_models

#--------------------------------------------------------
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r2001_e50_45_n8_r3 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r2001_e50_45_n8_r3 600 0.030 saved_models


#----
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" r902_e200_64_n5_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=5 sh run_eval_cnn.sh "5" r902_e200_64_n5_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r306_2nd_e50_49_n7r8 600 0.030 saved_models

#----------------------------------------------------------------
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_64_n5_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_64_n5_r1 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" r902_e200_64_n5_r1 600 0.030 saved_models

#----------------------------------------------------------------
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" V2_r2_e100 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" V2_r2_e100 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" V2_r3_e50 600 0.030 saved_models
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" V2_r3_e50 600 0.030 saved_models

#----------------------------------------------------------------
#CUDA_VISIBLE_DEVICES=2 sh run_eval_cnn.sh "2" r2001_e50_45_n8_r3 600 0.020 saved_models
#CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" r2001_e50_45_n8_r3 600 0.020 saved_models

#GDAS_MIXED_LEVEL1
##CUDA_VISIBLE_DEVICES=3 sh run_eval_cnn.sh "3" GDAS_MIXED_LEVEL1 6000 0.030 saved_models
##CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" GDAS_MIXED_LEVEL1 6001 0.030 saved_models


#GDAS_MIXED_LEVEL2
#CUDA_VISIBLE_DEVICES=0 sh run_eval_cnn.sh "0" GDAS_MIXED_LEVEL2 6000 0.030 saved_models
#CUDA_VISIBLE_DEVICES=1 sh run_eval_cnn.sh "1" GDAS_MIXED_LEVEL2 6001 0.030 saved_models

GPU=$1
ARCH=$2
EPOCH=$3
LR=$4
MODEL=$5

python evaluation/train.py \
--auxiliary \
--cutout \
--gpu $GPU \
--arch $ARCH \
--epochs $EPOCH \
--learning_rate $LR \
--model_path $MODEL