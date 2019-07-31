#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:54:57 2019

@author: xjc
"""


import os



for i in range(1, 83):
    orig_list_dir='./'+str(i) 
    train_file=orig_list_dir+'/train.txt'
    test_file=orig_list_dir+'/test.txt' 
    with open(train_file) as f:
        train_lines=f.readlines()
    with open(test_file) as ff:
        test_lines=ff.readlines()

    modified_train_file=orig_list_dir+'/modified_train.txt'
    modified_test_file=orig_list_dir+'/modified_test.txt'
    
    for file in [modified_train_file, modified_test_file]:
        if os.path.exists(file):
            os.remove(file)

##############generate new train list################
    for line in train_lines:
        img_name=line.split()[0]
        img_age=line.split()[1]
        new_age=''
        for i in range(1,101):
            if i<=int(img_age):
                new_age+=' 1'
            else:
                new_age+=' 0'
        new_record=img_name+new_age+' '+img_age+'\n'
        with open(modified_train_file, 'a') as ff:
            ff.write(new_record)

##############generate new test list#################### 
    for line in test_lines:
        img_name=line.split()[0]
        img_age=line.split()[1]
        new_age=''
        for i in range(1,101):
            if i<=int(img_age):
                new_age+=' 1'
            else:
                new_age+=' 0'
        new_record=img_name+new_age+' '+img_age+'\n'
        with open(modified_test_file, 'a') as ff:
            ff.write(new_record)
            




