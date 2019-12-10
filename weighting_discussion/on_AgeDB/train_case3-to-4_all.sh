#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case3_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case3_modified_cutdown_train_1.txt --validate_list case3_modified_validation_1.txt --test_list case3_modified_test_1.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case3_2.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case3_modified_cutdown_train_2.txt --validate_list case3_modified_validation_2.txt --test_list case3_modified_test_2.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case3_3.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case3_modified_cutdown_train_3.txt --validate_list case3_modified_validation_3.txt --test_list case3_modified_test_3.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case3_4.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case3_modified_cutdown_train_4.txt --validate_list case3_modified_validation_4.txt --test_list case3_modified_test_4.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case3_5.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case3_modified_cutdown_train_5.txt --validate_list case3_modified_validation_5.txt --test_list case3_modified_test_5.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date


date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case4_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case4_modified_cutdown_train_1.txt --validate_list case4_modified_validation_1.txt --test_list case4_modified_test_1.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case4_2.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case4_modified_cutdown_train_2.txt --validate_list case4_modified_validation_2.txt --test_list case4_modified_test_2.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case4_3.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case4_modified_cutdown_train_3.txt --validate_list case4_modified_validation_3.txt --test_list case4_modified_test_3.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case4_4.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case4_modified_cutdown_train_4.txt --validate_list case4_modified_validation_4.txt --test_list case4_modified_test_4.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0,1 python -u train_case4_5.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list case4_modified_cutdown_train_5.txt --validate_list case4_modified_validation_5.txt --test_list case4_modified_test_5.txt --save . --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date
