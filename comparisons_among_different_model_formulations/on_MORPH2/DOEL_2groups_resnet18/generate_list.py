#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:37:47 2019

@author: xjc
"""


import os

orig_train_list='./S1_train_processed.txt'
orig_valid_list='./S1_validation_processed.txt'
orig_test_list='./S1_test_processed.txt'

modified_train_list='./S1_train_modified.txt'
modified_valid_list='./S1_valid_modified.txt'
modified_test_list='./S1_test_modified.txt'

if os.path.exists(modified_train_list):
    os.remove(modified_train_list)
if os.path.exists(modified_valid_list):
    os.remove(modified_valid_list)
if os.path.exists(modified_test_list):
    os.remove(modified_test_list)
    
with open(orig_train_list) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,101):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list, 'a') as ff:
        ff.write(new_record)
        
        
        
with open(orig_valid_list) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,101):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list, 'a') as ff:
        ff.write(new_record)
        
        

with open(orig_test_list) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,101):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list, 'a') as ff:
        ff.write(new_record)
        
#################################
#################################
        
        
import os

orig_train_list='./S2_train_processed.txt'
orig_valid_list='./S2_validation_processed.txt'
orig_test_list='./S2_test_fixed_processed.txt'

modified_train_list='./S2_train_modified.txt'
modified_valid_list='./S2_valid_modified.txt'
modified_test_list='./S2_test_fixed_modified.txt'

if os.path.exists(modified_train_list):
    os.remove(modified_train_list)
if os.path.exists(modified_valid_list):
    os.remove(modified_valid_list)
if os.path.exists(modified_test_list):
    os.remove(modified_test_list)
    
with open(orig_train_list) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,101):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list, 'a') as ff:
        ff.write(new_record)
        
        
        
with open(orig_valid_list) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,101):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list, 'a') as ff:
        ff.write(new_record)
        
        

with open(orig_test_list) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,101):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list, 'a') as ff:
        ff.write(new_record)