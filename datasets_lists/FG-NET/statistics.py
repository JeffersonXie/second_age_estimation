#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 19:06:18 2019

@author: xjc
"""

import os

path='./images'
statistic='./imgs_information.txt'
files=os.listdir(path)

if os.path.exists(statistic):
    os.remove(statistic)
with open(statistic,'w') as f:
    f.write('img_name, img_id, img_age, img_gender:\n')    
    
for line in files:
    img_name=line
    img_id=''
    img_age=''
    flag=0
    for i in range(len(img_name)):
        if line[i].isdigit() and flag==0:
            img_id+=line[i]
        elif line[i].isdigit() and flag==1:
            img_age+=line[i]
        else:
            flag=1
    
    if img_id=='001':
        img_gender='M'
    if img_id=='002':
        img_gender='F'
    if img_id=='003':
        img_gender='F'
    if img_id=='004':
        img_gender='M'
    if img_id=='005':
        img_gender='F'
    if img_id=='006':
        img_gender='M'
    if img_id=='007':
        img_gender='M'
    if img_id=='008':
        img_gender='F'
    if img_id=='009':
        img_gender='F'
    if img_id=='010':
        img_gender='F'

    if img_id=='011':
        img_gender='M'
    if img_id=='012':
        img_gender='F'
    if img_id=='013':
        img_gender='M'
    if img_id=='014':
        img_gender='F'
    if img_id=='015':
        img_gender='F'
    if img_id=='016':
        img_gender='M'
    if img_id=='017':
        img_gender='M'
    if img_id=='018':
        img_gender='F'
    if img_id=='019':
        img_gender='M'
    if img_id=='020':
        img_gender='F'

    if img_id=='021':
        img_gender='F'
    if img_id=='022':
        img_gender='M'
    if img_id=='023':
        img_gender='M'
    if img_id=='024':
        img_gender='M'
    if img_id=='025':
        img_gender='F'
    if img_id=='026':
        img_gender='F'
    if img_id=='027':
        img_gender='F'
    if img_id=='028':
        img_gender='M'
    if img_id=='029':
        img_gender='M'
    if img_id=='030':
        img_gender='F'

    if img_id=='031':
        img_gender='M'
    if img_id=='032':
        img_gender='F'
    if img_id=='033':
        img_gender='M'
    if img_id=='034':
        img_gender='F'
    if img_id=='035':
        img_gender='M'
    if img_id=='036':
        img_gender='M'
    if img_id=='037':
        img_gender='M'
    if img_id=='038':
        img_gender='M'
    if img_id=='039':
        img_gender='F'
    if img_id=='040':
        img_gender='M'

    if img_id=='041':
        img_gender='M'
    if img_id=='042':
        img_gender='M'
    if img_id=='043':
        img_gender='F'
    if img_id=='044':
        img_gender='M'
    if img_id=='045':
        img_gender='M'
    if img_id=='046':
        img_gender='M'
    if img_id=='047':
        img_gender='F'
    if img_id=='048':
        img_gender='M'
    if img_id=='049':
        img_gender='F'
    if img_id=='050':
        img_gender='M'

    if img_id=='051':
        img_gender='M'
    if img_id=='052':
        img_gender='F'
    if img_id=='053':
        img_gender='M'
    if img_id=='054':
        img_gender='F'
    if img_id=='055':
        img_gender='M'
    if img_id=='056':
        img_gender='M'
    if img_id=='057':
        img_gender='M'
    if img_id=='058':
        img_gender='M'
    if img_id=='059':
        img_gender='F'
    if img_id=='060':
        img_gender='F'

    if img_id=='061':
        img_gender='F'
    if img_id=='062':
        img_gender='F'
    if img_id=='063':
        img_gender='M'
    if img_id=='064':
        img_gender='M'
    if img_id=='065':
        img_gender='F'
    if img_id=='066':
        img_gender='M'
    if img_id=='067':
        img_gender='F'
    if img_id=='068':
        img_gender='M'
    if img_id=='069':
        img_gender='M'
    if img_id=='070':
        img_gender='M'

    if img_id=='071':
        img_gender='M'
    if img_id=='072':
        img_gender='F'
    if img_id=='073':
        img_gender='F'
    if img_id=='074':
        img_gender='M'
    if img_id=='075':
        img_gender='M'
    if img_id=='076':
        img_gender='F'
    if img_id=='077':
        img_gender='F'
    if img_id=='078':
        img_gender='M'
    if img_id=='079':
        img_gender='M'
    if img_id=='080':
        img_gender='M'

    if img_id=='081':
        img_gender='M'
    if img_id=='082':
        img_gender='M'
      
        
        
        
    record=img_name+' '+img_id+' '+img_age+' '+img_gender+'\n'
    with open(statistic,'a') as f:
        f.write(record)
    