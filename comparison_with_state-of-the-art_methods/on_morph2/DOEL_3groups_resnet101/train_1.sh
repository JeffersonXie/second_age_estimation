#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
CUDA_VISIBLE_DEVICES=0 python -u train_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S1_full_train_probs_1.txt --test_list S1_test_probs_1.txt --save /home/xjc/test/ensemble_learning_age_estimation/on_morph2 --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date






