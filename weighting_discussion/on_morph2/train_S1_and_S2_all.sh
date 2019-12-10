#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
date
CUDA_VISIBLE_DEVICES=0,1 python -u train_S1_case1_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S1_train_probs_case1.txt --validate_list S1_valid_probs_case1.txt --test_list S1_test_probs_case1.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_S1_case3_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S1_train_probs_case3.txt --validate_list S1_valid_probs_case3.txt --test_list S1_test_probs_case3.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_S1_case4_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S1_train_probs_case4.txt --validate_list S1_valid_probs_case4.txt --test_list S1_test_probs_case4.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_S2_case1_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S2_train_probs_case1.txt --validate_list S2_valid_probs_case1.txt --test_list S2_test_fixed_probs_case1.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_S2_case3_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S2_train_probs_case3.txt --validate_list S2_valid_probs_case3.txt --test_list S2_test_fixed_probs_case3.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_S2_case4_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --train_list S2_train_probs_case4.txt --validate_list S2_valid_probs_case4.txt --test_list S2_test_fixed_probs_case4.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date





