#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 10:02:49 2019

@author: xjc
"""
import os
import shutil

imgs_information_dir='./imgs_information.txt'
save_path_dir='./leave_one_person_out'
if os.path.exists(save_path_dir):
    shutil.rmtree(save_path_dir)
os.mkdir(save_path_dir)

for i in range(1,83):
    tmp_dir=save_path_dir+'/'+str(i)
    os.mkdir(tmp_dir)
    tmp_train_list_dir=tmp_dir+'/'+'train.txt'
    tmp_test_list_dir=tmp_dir+'/'+'test.txt'
    with open(tmp_train_list_dir,'w') as f:
        f.write('img_name, img_id, img_age, img_gender:\n')
    with open(tmp_test_list_dir,'w') as f:
        f.write('img_name, img_id, img_age, img_gender:\n')
        
    with open(imgs_information_dir) as f:
        lines=f.readlines()
    lines.pop(0)
    for line in lines:
        img_id=line.split()[1]
        if int(img_id)==i:
            with open(tmp_test_list_dir,'a') as f:
                f.write(line)
        else:
            with open(tmp_train_list_dir,'a') as f:
                f.write(line)