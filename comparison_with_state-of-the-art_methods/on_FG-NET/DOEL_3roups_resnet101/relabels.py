#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:17:56 2019

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
        with open(modified_train_file, 'a') as ff:
            ff.write(tmp_record)



##############generate new test list####################            
    for line in test_lines:
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
        with open(modified_test_file, 'a') as ff:
            ff.write(tmp_record)




