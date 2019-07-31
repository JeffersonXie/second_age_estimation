#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
CUDA_VISIBLE_DEVICES=0 python -u train_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_FG_NET_version_2 --save /home/xjc/test/ensemble_learning_age_estimation/on_FG_NET/two_group --n_epochs 1 --batch_size 32 --lr 0.001 --seed 2
date






