#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:00:05 2019

@author: xjc
"""



import os

path='./AgeDB'
statistic='./imgs_information.txt'
files=os.listdir(path)

if os.path.exists(statistic):
    os.remove(statistic)
with open(statistic,'w') as f:
    f.write('img_name, img_age, img_gender:\n')    
    
for line in files:
    img_name=line
    img_age=img_name.split('_')[2]
    img_gender=img_name.split('_')[-1].split('.')[0]
    
    if img_gender=='f':
        record=img_name+' '+img_age+' F'+'\n'
    else:
        record=img_name+' '+img_age+' M'+'\n'

    with open(statistic,'a') as f:
        f.write(record)
    