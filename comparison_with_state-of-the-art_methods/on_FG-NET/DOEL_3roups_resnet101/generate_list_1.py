#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:41:07 2019

@author: xjc
"""


import os
import shutil
import random

orig_lists_dir='/media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related(first_paper)/original_dataset/FG-NET/leave_one_person_out'
save_lists_dir='./'
for i in range(1, 83):
    orig_list_dir=orig_lists_dir+'/'+str(i) 
    train_file=orig_list_dir+'/train.txt'
    test_file=orig_list_dir+'/test.txt' 
    with open(train_file) as f:
        train_lines=f.readlines()
    train_lines.pop(0)
    random.shuffle(train_lines)
    with open(test_file) as ff:
        test_lines=ff.readlines()
    test_lines.pop(0)
    
#    save_file_dir=save_lists_dir+'/'+str(i)
    save_file_dir=save_lists_dir+str(i)
    if os.path.exists(save_file_dir):  
        shutil.rmtree(save_file_dir)
    os.mkdir(save_file_dir)
    
    save_train_file=save_file_dir+'/train.txt'
    save_train_file_tmp=open(save_train_file,'w+')
    for train_line in train_lines:
        img_name=train_line.split()[0]
        img_age=train_line.split()[2]
        img_gender=train_line.split()[3]
        if img_gender=='M':
            img_gender='0'
        else:
            img_gender='1'
        processed_train_line=img_name+' '+str(int(img_age))+'\n'
        save_train_file_tmp.write(processed_train_line)
    save_train_file_tmp.close()
    

    save_test_file=save_file_dir+'/test.txt'
    save_test_file_tmp=open(save_test_file,'w+')
    for test_line in test_lines:
        img_name=test_line.split()[0]
        img_age=test_line.split()[2]
        img_gender=test_line.split()[3]
        if img_gender=='M':
            img_gender='0'
        else:
            img_gender='1'
        processed_test_line=img_name+' '+str(int(img_age))+'\n'
        save_test_file_tmp.write(processed_test_line)
    save_test_file_tmp.close()
    
    del orig_list_dir, f, ff, train_lines, test_lines
    