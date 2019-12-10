#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:40:46 2019

@author: xjc
"""

#import os

orig_train_file='/media/xjc/C14D581BDA18EBFA/ChaLearn LAP 2015/Train.csv'
orig_valid_file='/media/xjc/C14D581BDA18EBFA/ChaLearn LAP 2015/Validation.csv'
orig_test_file='/media/xjc/C14D581BDA18EBFA/ChaLearn LAP 2015/Test.csv'
orig_train_plus_valid_file='/media/xjc/C14D581BDA18EBFA/ChaLearn LAP 2015/Train_plus_validation.csv'

new_train_file='./train_probs_two_group.txt'
new_valid_file='./valid_probs_two_group.txt'
new_test_file='./test_probs_two_group.txt'
new_train_plus_valid_file='./train_plus_valid_probs_two_group.txt'





####for train list ########
with open(orig_train_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split(';')[0]
    img_mean_age=int(line.split(';')[1])
    img_standard_deviation=line.split(';')[2].split()[0]
    new_age=''
    for i in range(1,101):
        if i<=img_mean_age:
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+str(img_mean_age)+' '+img_standard_deviation+'\n'
    with open(new_train_file, 'a') as ff:
        ff.write(new_record)
        
        
        
####for valid list ########
with open(orig_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split(';')[0]
    img_mean_age=int(line.split(';')[1])
    img_standard_deviation=line.split(';')[2].split()[0]
    new_age=''
    for i in range(1,101):
        if i<=img_mean_age:
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+str(img_mean_age)+' '+img_standard_deviation+'\n'
    with open(new_valid_file, 'a') as ff:
        ff.write(new_record)




####for test list ########
with open(orig_test_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split(';')[0]
    img_mean_age=int(line.split(';')[1])
    img_standard_deviation=line.split(';')[2].split()[0]
    new_age=''
    for i in range(1,101):
        if i<=img_mean_age:
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+str(img_mean_age)+' '+img_standard_deviation+'\n'
    with open(new_test_file, 'a') as ff:
        ff.write(new_record)


        
#####for train_plus_valid lsit#########################
with open(orig_train_plus_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split(';')[0]
    img_mean_age=int(line.split(';')[1])
    img_standard_deviation=line.split(';')[2].split()[0]
    new_age=''
    for i in range(1,101):
        if i<=img_mean_age:
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+str(img_mean_age)+' '+img_standard_deviation+'\n'
    with open(new_train_plus_valid_file, 'a') as ff:
        ff.write(new_record)