#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:36:59 2019

@author: xjc
"""


import os
import math

#import numpy as np 
#import matplotlib.pyplot as plt 


sigma=3

orig_train_list_1='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_1.txt'
orig_valid_list_1='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_1.txt'
orig_test_list_1='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_1.txt'

orig_train_list_2='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_2.txt'
orig_valid_list_2='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_2.txt'
orig_test_list_2='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_2.txt'

orig_train_list_3='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_3.txt'
orig_valid_list_3='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_3.txt'
orig_test_list_3='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_3.txt'

orig_train_list_4='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_4.txt'
orig_valid_list_4='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_4.txt'
orig_test_list_4='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_4.txt'

orig_train_list_5='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_cutdown_train_5.txt'
orig_valid_list_5='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_validation_5.txt'
orig_test_list_5='/home/xjc/test/comparisons_of_different_losses/on_AgeDB/five_fold_cross_validation/orig_test_5.txt'



modified_train_list_1='./processed_cutdown_train_1.txt'
modified_valid_list_1='./processed_validation_1.txt'
modified_test_list_1='./processed_test_1.txt'

modified_train_list_2='./processed_cutdown_train_2.txt'
modified_valid_list_2='./processed_validation_2.txt'
modified_test_list_2='./processed_test_2.txt'

modified_train_list_3='./processed_cutdown_train_3.txt'
modified_valid_list_3='./processed_validation_3.txt'
modified_test_list_3='./processed_test_3.txt'

modified_train_list_4='./processed_cutdown_train_4.txt'
modified_valid_list_4='./processed_validation_4.txt'
modified_test_list_4='./processed_test_4.txt'

modified_train_list_5='./processed_cutdown_train_5.txt'
modified_valid_list_5='./processed_validation_5.txt'
modified_test_list_5='./processed_test_5.txt'



for tmp_file in [modified_train_list_1, modified_valid_list_1, modified_test_list_1]:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        
for tmp_file in [modified_train_list_2, modified_valid_list_2, modified_test_list_2]:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        
for tmp_file in [modified_train_list_3, modified_valid_list_3, modified_test_list_3]:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        
for tmp_file in [modified_train_list_4, modified_valid_list_4, modified_test_list_4]:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        
for tmp_file in [modified_train_list_5, modified_valid_list_5, modified_test_list_5]:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

###############################################
############*******************################
###############################################

################################################
with open(orig_train_list_1) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_train_list_1, 'a') as ff:
        ff.write(new_record)


with open(orig_valid_list_1) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_valid_list_1, 'a') as ff:
        ff.write(new_record)
#        
        

with open(orig_test_list_1) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_test_list_1, 'a') as ff:
        ff.write(new_record)
###############################################
with open(orig_train_list_2) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_train_list_2, 'a') as ff:
        ff.write(new_record)


with open(orig_valid_list_2) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_valid_list_2, 'a') as ff:
        ff.write(new_record)
#        
        

with open(orig_test_list_2) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_test_list_2, 'a') as ff:
        ff.write(new_record)
###############################################
with open(orig_train_list_3) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_train_list_3, 'a') as ff:
        ff.write(new_record)


with open(orig_valid_list_3) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_valid_list_3, 'a') as ff:
        ff.write(new_record)
#        
        

with open(orig_test_list_3) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_test_list_3, 'a') as ff:
        ff.write(new_record)
###############################################
with open(orig_train_list_4) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_train_list_4, 'a') as ff:
        ff.write(new_record)


with open(orig_valid_list_4) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_valid_list_4, 'a') as ff:
        ff.write(new_record)
#        
        

with open(orig_test_list_4) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_test_list_4, 'a') as ff:
        ff.write(new_record)
###############################################
with open(orig_train_list_5) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_train_list_5, 'a') as ff:
        ff.write(new_record)


with open(orig_valid_list_5) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_valid_list_5, 'a') as ff:
        ff.write(new_record)
#        
        

with open(orig_test_list_5) as f:
    records=f.readlines()
for record in records:    
    img_name=record.split()[0]
    img_age=int(record.split()[1])
    classes_prob=[]
    for i in range(0,102):
        temp_class_prob=1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(i-img_age)**2/(2*sigma**2))
        classes_prob.append(temp_class_prob)    
    sum_classes_prob=sum(classes_prob)
    new_classes_prob=[x/sum_classes_prob for x in classes_prob]

    new_age=''
    for i in range(0,102):
        new_age+=' '+str(round(new_classes_prob[i],4))   
    new_record=img_name+new_age+' '+str(img_age)+'\n'
    with open(modified_test_list_5, 'a') as ff:
        ff.write(new_record)
###############################################