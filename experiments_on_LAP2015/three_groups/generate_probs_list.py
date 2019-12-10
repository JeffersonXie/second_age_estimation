#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:22:32 2019

@author: xjc
"""




import os
#import math

orig_train_plus_valid_file='/media/xjc/C14D581BDA18EBFA/ChaLearn LAP 2015/Train_plus_validation.csv'
orig_test_file='/media/xjc/C14D581BDA18EBFA/ChaLearn LAP 2015/Test.csv'


new_train_plus_valid_file='./train_plus_valid_probs_three_group.txt'
new_test_file='./test_probs_three_group.txt'


if os.path.exists(new_train_plus_valid_file):
    os.remove(new_train_plus_valid_file)
if os.path.exists(new_test_file):
    os.remove(new_test_file)



sigma=2


#####for train_plus_valid lsit#########################
with open(orig_train_plus_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split(';')[0]
    img_mean_age=int(line.split(';')[1])
    img_standard_deviation=line.split(';')[2].split()[0]
    img_label=img_name
    for i in range(1,100):
        if i<img_mean_age:    
            prob_g1=0.0
            prob_g2=0.2
            prob_g3=0.8
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_mean_age:
            prob_g1=0.1
            prob_g2=0.8
            prob_g3=0.1
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.8
            prob_g2=0.2
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_mean_age)+' '+img_standard_deviation+'\n'
    with open(new_train_plus_valid_file, 'a') as ff:
        ff.write(tmp_record)



####for test list ########
with open(orig_test_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split(';')[0]
    img_mean_age=int(line.split(';')[1])
    img_standard_deviation=line.split(';')[2].split()[0]
    img_label=img_name
    for i in range(1,100):
        if i<img_mean_age:    
            prob_g1=0.0
            prob_g2=0.2
            prob_g3=0.8
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_mean_age:
            prob_g1=0.1
            prob_g2=0.8
            prob_g3=0.1
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.8
            prob_g2=0.2
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_mean_age)+' '+img_standard_deviation+'\n'
    with open(new_test_file, 'a') as ff:
        ff.write(tmp_record)
        