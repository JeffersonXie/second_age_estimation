#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:05:21 2019

@author: xjc
"""





import os

orig_train_list_1='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_1.txt'
orig_valid_list_1='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_1.txt'
orig_test_list_1='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_1.txt'

modified_train_list_1='./case3_modified_cutdown_train_1.txt'
modified_valid_list_1='./case3_modified_validation_1.txt'
modified_test_list_1='./case3_modified_test_1.txt'

orig_train_list_2='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_2.txt'
orig_valid_list_2='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_2.txt'
orig_test_list_2='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_2.txt'

modified_train_list_2='./case3_modified_cutdown_train_2.txt'
modified_valid_list_2='./case3_modified_validation_2.txt'
modified_test_list_2='./case3_modified_test_2.txt'

orig_train_list_3='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_3.txt'
orig_valid_list_3='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_3.txt'
orig_test_list_3='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_3.txt'

modified_train_list_3='./case3_modified_cutdown_train_3.txt'
modified_valid_list_3='./case3_modified_validation_3.txt'
modified_test_list_3='./case3_modified_test_3.txt'

orig_train_list_4='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_4.txt'
orig_valid_list_4='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_4.txt'
orig_test_list_4='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_4.txt'

modified_train_list_4='./case3_modified_cutdown_train_4.txt'
modified_valid_list_4='./case3_modified_validation_4.txt'
modified_test_list_4='./case3_modified_test_4.txt'

orig_train_list_5='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_5.txt'
orig_valid_list_5='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_5.txt'
orig_test_list_5='/home/xjc/test/experiments-for-fisrt-submission/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_5.txt'

modified_train_list_5='./case3_modified_cutdown_train_5.txt'
modified_valid_list_5='./case3_modified_validation_5.txt'
modified_test_list_5='./case3_modified_test_5.txt'


#
#if os.path.exists(new_train_file):
#    os.remove(new_train_file)
#if os.path.exists(new_valid_file):
#    os.remove(new_valid_file)
#if os.path.exists(new_test_file):
#    os.remove(new_test_file)


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



sigma=2


#####for train lsit#########################
with open(orig_train_list_1) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_train_list_1, 'a') as ff:
        ff.write(tmp_record)

        
with open(orig_train_list_2) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_train_list_2, 'a') as ff:
        ff.write(tmp_record)


with open(orig_train_list_3) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_train_list_3, 'a') as ff:
        ff.write(tmp_record)

        
with open(orig_train_list_4) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_train_list_4, 'a') as ff:
        ff.write(tmp_record)

        
with open(orig_train_list_5) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_train_list_5, 'a') as ff:
        ff.write(tmp_record)



####for valid list ########
with open(orig_valid_list_1) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_valid_list_1, 'a') as ff:
        ff.write(tmp_record)
  
      
with open(orig_valid_list_2) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_valid_list_2, 'a') as ff:
        ff.write(tmp_record)
    
    
with open(orig_valid_list_3) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_valid_list_3, 'a') as ff:
        ff.write(tmp_record)
        
        
with open(orig_valid_list_4) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_valid_list_4, 'a') as ff:
        ff.write(tmp_record)
        
        
with open(orig_valid_list_5) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_valid_list_5, 'a') as ff:
        ff.write(tmp_record)

#######for test list#################
with open(orig_test_list_1) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_test_list_1, 'a') as ff:
        ff.write(tmp_record)
    
    
with open(orig_test_list_2) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_test_list_2, 'a') as ff:
        ff.write(tmp_record)
       
        
with open(orig_test_list_3) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_test_list_3, 'a') as ff:
        ff.write(tmp_record)
        
        
with open(orig_test_list_4) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_test_list_4, 'a') as ff:
        ff.write(tmp_record)
        
        
with open(orig_test_list_5) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,101):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.3
            prob_g3=0.7
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.15
            prob_g2=0.7
            prob_g3=0.15
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.7
            prob_g2=0.3
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(modified_test_list_5, 'a') as ff:
        ff.write(tmp_record)