#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_11_1.py --data_root /media/xjc/C14D581BDA18EBFA/ChaLearn\ LAP\ 2015/LAP2015_aligned_all_by_binbingao --train_list ./train_plus_valid_probs_three_group.txt --test_list ./test_probs_three_group.txt --save . --n_epochs 65 --batch_size 32 --lr 0.0001 --seed 2
date






