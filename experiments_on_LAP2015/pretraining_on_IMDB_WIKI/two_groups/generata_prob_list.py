#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:47:29 2019

@author: xjc
"""






import os
import math

orig_train_file='/home/xjc/test/experiments-for-second-submission (major-revision)/experiments-on-IMDB-WIKI/train.txt'
orig_valid_file='/home/xjc/test/experiments-for-second-submission (major-revision)/experiments-on-IMDB-WIKI/validation.txt'


new_train_file='./train_probs_two_group.txt'
new_valid_file='./valid_probs_two_group.txt'


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
    new_age=''
    for i in range(1,101):
        if i<=img_age:
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(new_train_file, 'a') as ff:
        ff.write(new_record)




####for valid list ########
with open(orig_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    new_age=''
    for i in range(1,101):
        if i<=img_age:
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(new_valid_file, 'a') as ff:
        ff.write(new_record)
