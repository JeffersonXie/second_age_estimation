#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:34:17 2019

@author: xjc
"""
import os

orig_S1_train_prob_file='/home/xjc/test/ensemble_learning_age_estimation/try_on_morph2/resnet18_version/three_group/S1_train_probs_1.txt'
orig_S1_valid_prob_file='/home/xjc/test/ensemble_learning_age_estimation/try_on_morph2/resnet18_version/three_group/S1_valid_probs_1.txt'

orig_S2_train_prob_file='/home/xjc/test/ensemble_learning_age_estimation/try_on_morph2/resnet18_version/three_group/S2_train_probs_1.txt'
orig_S2_valid_prob_file='/home/xjc/test/ensemble_learning_age_estimation/try_on_morph2/resnet18_version/three_group/S2_valid_probs_1.txt'

full_S1_train_prob_file='./S1_full_train_probs_1.txt'
full_S2_train_prob_file='./S2_full_train_probs_1.txt'

if os.path.exists(full_S1_train_prob_file):
    os.remove(full_S1_train_prob_file)
if os.path.exists(full_S2_train_prob_file):
    os.remove(full_S2_train_prob_file)


with open(orig_S1_train_prob_file) as f:
    lines_1=f.readlines()
with open(orig_S1_valid_prob_file) as f:
    lines_2=f.readlines()
lines=lines_1+lines_2
for line in lines:
    with open(full_S1_train_prob_file, 'a') as f:
        f.write(line)
       
        
        
        
with open(orig_S2_train_prob_file) as f:
    lines_1=f.readlines()
with open(orig_S2_valid_prob_file) as f:
    lines_2=f.readlines()
lines=lines_1+lines_2
for line in lines:
    with open(full_S2_train_prob_file, 'a') as f:
        f.write(line)