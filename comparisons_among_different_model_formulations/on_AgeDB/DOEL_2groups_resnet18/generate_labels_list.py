#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:16:04 2019

@author: xjc
"""



import os

orig_train_list_1='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_1.txt'
orig_valid_list_1='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_1.txt'
orig_test_list_1='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_1.txt'

modified_train_list_1='./processed_cutdown_train_1.txt'
modified_valid_list_1='./processed_validation_1.txt'
modified_test_list_1='./processed_test_1.txt'

orig_train_list_2='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_2.txt'
orig_valid_list_2='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_2.txt'
orig_test_list_2='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_2.txt'

modified_train_list_2='./processed_cutdown_train_2.txt'
modified_valid_list_2='./processed_validation_2.txt'
modified_test_list_2='./processed_test_2.txt'

orig_train_list_3='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_3.txt'
orig_valid_list_3='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_3.txt'
orig_test_list_3='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_3.txt'

modified_train_list_3='./processed_cutdown_train_3.txt'
modified_valid_list_3='./processed_validation_3.txt'
modified_test_list_3='./processed_test_3.txt'

orig_train_list_4='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_4.txt'
orig_valid_list_4='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_4.txt'
orig_test_list_4='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_4.txt'

modified_train_list_4='./processed_cutdown_train_4.txt'
modified_valid_list_4='./processed_validation_4.txt'
modified_test_list_4='./processed_test_4.txt'

orig_train_list_5='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_5.txt'
orig_valid_list_5='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_5.txt'
orig_test_list_5='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_5.txt'

modified_train_list_5='./processed_cutdown_train_5.txt'
modified_valid_list_5='./processed_validation_5.txt'
modified_test_list_5='./processed_test_5.txt'


if os.path.exists(modified_train_list_1):
    os.remove(modified_train_list_1)
if os.path.exists(modified_valid_list_1):
    os.remove(modified_valid_list_1)
if os.path.exists(modified_test_list_1):
    os.remove(modified_test_list_1)

if os.path.exists(modified_train_list_2):
    os.remove(modified_train_list_2)
if os.path.exists(modified_valid_list_2):
    os.remove(modified_valid_list_2)
if os.path.exists(modified_test_list_2):
    os.remove(modified_test_list_2)
    
if os.path.exists(modified_train_list_3):
    os.remove(modified_train_list_3)
if os.path.exists(modified_valid_list_3):
    os.remove(modified_valid_list_3)
if os.path.exists(modified_test_list_3):
    os.remove(modified_test_list_3)
    
if os.path.exists(modified_train_list_4):
    os.remove(modified_train_list_4)
if os.path.exists(modified_valid_list_4):
    os.remove(modified_valid_list_4)
if os.path.exists(modified_test_list_4):
    os.remove(modified_test_list_4)
    
if os.path.exists(modified_train_list_5):
    os.remove(modified_train_list_5)
if os.path.exists(modified_valid_list_5):
    os.remove(modified_valid_list_5)
if os.path.exists(modified_test_list_5):
    os.remove(modified_test_list_5)



################################################################    
with open(orig_train_list_1) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list_1, 'a') as ff:
        ff.write(new_record)
        
with open(orig_valid_list_1) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list_1, 'a') as ff:
        ff.write(new_record)

with open(orig_test_list_1) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list_1, 'a') as ff:
        ff.write(new_record)
#####################################################
with open(orig_train_list_2) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list_2, 'a') as ff:
        ff.write(new_record)
        
with open(orig_valid_list_2) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list_2, 'a') as ff:
        ff.write(new_record)

with open(orig_test_list_2) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list_2, 'a') as ff:
        ff.write(new_record)
#####################################################
with open(orig_train_list_3) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list_3, 'a') as ff:
        ff.write(new_record)
        
with open(orig_valid_list_3) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list_3, 'a') as ff:
        ff.write(new_record)

with open(orig_test_list_3) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list_3, 'a') as ff:
        ff.write(new_record)
#####################################################
with open(orig_train_list_4) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list_4, 'a') as ff:
        ff.write(new_record)
        
with open(orig_valid_list_4) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list_4, 'a') as ff:
        ff.write(new_record)

with open(orig_test_list_4) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list_4, 'a') as ff:
        ff.write(new_record)
##########################################################
with open(orig_train_list_5) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_train_list_5, 'a') as ff:
        ff.write(new_record)
        
with open(orig_valid_list_5) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_valid_list_5, 'a') as ff:
        ff.write(new_record)

with open(orig_test_list_5) as f:
    records=f.readlines()
for record in records:
    img_name=record.split()[0]
    img_age=record.split()[1]
    new_age=''
    for i in range(1,102):
        if i<=int(img_age):
            new_age+=' 1'
        else:
            new_age+=' 0'
    new_record=img_name+new_age+' '+img_age+'\n'
    with open(modified_test_list_5, 'a') as ff:
        ff.write(new_record)        