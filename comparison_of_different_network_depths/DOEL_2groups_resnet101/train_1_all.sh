#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
date
CUDA_VISIBLE_DEVICES=0 python -u train_1_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list processed_cutdown_train_1.txt --validate_list processed_validation_1.txt --test_list processed_test_1.txt --save /home/xjc/test/ensemble_learning_age_estimation/idea_try_1/try_on_AgeDB/resnet101_version/two_group --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0 python -u train_1_2.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list processed_cutdown_train_2.txt --validate_list processed_validation_2.txt --test_list processed_test_2.txt --save /home/xjc/test/ensemble_learning_age_estimation/idea_try_1/try_on_AgeDB/resnet101_version/two_group --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0 python -u train_1_3.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list processed_cutdown_train_3.txt --validate_list processed_validation_3.txt --test_list processed_test_3.txt --save /home/xjc/test/ensemble_learning_age_estimation/idea_try_1/try_on_AgeDB/resnet101_version/two_group --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0 python -u train_1_4.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list processed_cutdown_train_4.txt --validate_list processed_validation_4.txt --test_list processed_test_4.txt --save /home/xjc/test/ensemble_learning_age_estimation/idea_try_1/try_on_AgeDB/resnet101_version/two_group --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0 python -u train_1_5.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list processed_cutdown_train_5.txt --validate_list processed_validation_5.txt --test_list processed_test_5.txt --save /home/xjc/test/ensemble_learning_age_estimation/idea_try_1/try_on_AgeDB/resnet101_version/two_group --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

