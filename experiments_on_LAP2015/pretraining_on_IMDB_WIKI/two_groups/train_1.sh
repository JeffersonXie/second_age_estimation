#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_1.py --data_root /media/xjc/C14D581BDA18EBFA/IMDB-WIKI --train_list ./train_probs_two_group.txt --validate_list ./valid_probs_two_group.txt --save . --n_epochs 65 --batch_size 64 --lr 0.001 --seed 2
date






