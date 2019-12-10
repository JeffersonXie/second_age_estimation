#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:22:48 2019

@author: xjc
"""




import os
import math

orig_train_file='/home/xjc/test/experiments-for-second-submission (major-revision)/experiments-on-IMDB-WIKI/train.txt'
orig_valid_file='/home/xjc/test/experiments-for-second-submission (major-revision)/experiments-on-IMDB-WIKI/validation.txt'


new_train_file='./train_probs_three_group.txt'
new_valid_file='./valid_probs_three_group.txt'


if os.path.exists(new_train_file):
    os.remove(new_train_file)
if os.path.exists(new_valid_file):
    os.remove(new_valid_file)



sigma=2


#####for train lsit#########################
with open(orig_train_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.2
            prob_g3=0.8
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
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
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_train_file, 'a') as ff:
        ff.write(tmp_record)



####for valid list ########
with open(orig_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.2
            prob_g3=0.8
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
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
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_valid_file, 'a') as ff:
        ff.write(tmp_record)
        
