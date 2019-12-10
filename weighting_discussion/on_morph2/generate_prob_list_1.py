#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:23:52 2019

@author: xjc
"""




import os
import math

orig_S1_train_file='./S1_train_processed.txt'
orig_S1_valid_file='./S1_validation_processed.txt'
orig_S1_test_file='./S1_test_processed.txt'

new_S1_train_file_case1='./S1_train_probs_case1.txt'
new_S1_valid_file_case1='./S1_valid_probs_case1.txt'
new_S1_test_file_case1='./S1_test_probs_case1.txt'


new_S1_train_file_case3='./S1_train_probs_case3.txt'
new_S1_valid_file_case3='./S1_valid_probs_case3.txt'
new_S1_test_file_case3='./S1_test_probs_case3.txt'


new_S1_train_file_case4='./S1_train_probs_case4.txt'
new_S1_valid_file_case4='./S1_valid_probs_case4.txt'
new_S1_test_file_case4='./S1_test_probs_case4.txt'


if os.path.exists(new_S1_train_file_case1):
    os.remove(new_S1_train_file_case1)
if os.path.exists(new_S1_valid_file_case1):
    os.remove(new_S1_valid_file_case1)
if os.path.exists(new_S1_test_file_case1):
    os.remove(new_S1_test_file_case1)
    
    
if os.path.exists(new_S1_train_file_case3):
    os.remove(new_S1_train_file_case3)
if os.path.exists(new_S1_valid_file_case3):
    os.remove(new_S1_valid_file_case3)
if os.path.exists(new_S1_test_file_case3):
    os.remove(new_S1_test_file_case3)
    
    
if os.path.exists(new_S1_train_file_case4):
    os.remove(new_S1_train_file_case4)
if os.path.exists(new_S1_valid_file_case4):
    os.remove(new_S1_valid_file_case4)
if os.path.exists(new_S1_test_file_case4):
    os.remove(new_S1_test_file_case4)




#####for train lsit#########################
with open(orig_S1_train_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.1
            prob_g3=0.9
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.05
            prob_g2=0.9
            prob_g3=0.05
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.9
            prob_g2=0.1
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_S1_train_file_case1, 'a') as ff:
        ff.write(tmp_record)
        
        
with open(orig_S1_train_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
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
    with open(new_S1_train_file_case3, 'a') as ff:
        ff.write(tmp_record)
        
        
        
with open(orig_S1_train_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.4
            prob_g3=0.6
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.2
            prob_g2=0.6
            prob_g3=0.2
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.6
            prob_g2=0.4
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_S1_train_file_case4, 'a') as ff:
        ff.write(tmp_record)



####for valid list ########
with open(orig_S1_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.1
            prob_g3=0.9
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.05
            prob_g2=0.9
            prob_g3=0.05
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.9
            prob_g2=0.1
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_S1_valid_file_case1, 'a') as ff:
        ff.write(tmp_record)
        
        

with open(orig_S1_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
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
    with open(new_S1_valid_file_case3, 'a') as ff:
        ff.write(tmp_record)
        
        
        
with open(orig_S1_valid_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.4
            prob_g3=0.6
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.2
            prob_g2=0.6
            prob_g3=0.2
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.6
            prob_g2=0.4
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_S1_valid_file_case4, 'a') as ff:
        ff.write(tmp_record)
        
        
        
        
        
#######for test list#################
with open(orig_S1_test_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.1
            prob_g3=0.9
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.05
            prob_g2=0.9
            prob_g3=0.05
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.9
            prob_g2=0.1
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_S1_test_file_case1, 'a') as ff:
        ff.write(tmp_record)
        
        
        
with open(orig_S1_test_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
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
    with open(new_S1_test_file_case3, 'a') as ff:
        ff.write(tmp_record)
        
        
        
with open(orig_S1_test_file) as f:
    lines=f.readlines()
for line in lines:
    img_name=line.split()[0]
    img_age=int(line.split()[1])
    img_label=img_name
    for i in range(1,100):
        if i<img_age:    
            prob_g1=0.0
            prob_g2=0.4
            prob_g3=0.6
            tmp_class=2
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        elif i==img_age:
            prob_g1=0.2
            prob_g2=0.6
            prob_g3=0.2
            tmp_class=1
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
        else:            
            prob_g1=0.6
            prob_g2=0.4
            prob_g3=0.0
            tmp_class=0
            img_label+=' '+str(prob_g1)+' '+str(prob_g2)+' '+str(prob_g3)+' '+str(tmp_class)
    tmp_record=img_label+' '+str(img_age)+'\n'
    with open(new_S1_test_file_case4, 'a') as ff:
        ff.write(tmp_record)