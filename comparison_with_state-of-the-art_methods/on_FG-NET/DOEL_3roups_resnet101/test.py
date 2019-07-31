#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:01:50 2019

@author: xjc
"""

orig_file='./el_resnet101_nesterov_results_1.csv'

with open (orig_file) as f:
    lines=f.readlines()

for i in range(1,5916):
    lines.pop(0)

for i in range(1,5):
    lines.pop(-1)
    
    
CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
for line in lines:
    img_age_error=int(line.split()[-1])
    
    if img_age_error<=1:
        CS_1_numerator+=1
    if img_age_error<=2:
        CS_2_numerator+=1
    if img_age_error<=3:
        CS_3_numerator+=1
    if img_age_error<=4:
        CS_4_numerator+=1
    if img_age_error<=5:
        CS_5_numerator+=1
    if img_age_error<=6:
        CS_6_numerator+=1
    if img_age_error<=7:
        CS_7_numerator+=1
    if img_age_error<=8:
        CS_8_numerator+=1
    if img_age_error<=9:
        CS_9_numerator+=1
    if img_age_error<=10:
        CS_10_numerator+=1
        
CS_1=CS_1_numerator/len(lines)
CS_2=CS_2_numerator/len(lines)
CS_3=CS_3_numerator/len(lines)
CS_4=CS_4_numerator/len(lines)
CS_5=CS_5_numerator/len(lines)
CS_6=CS_6_numerator/len(lines)
CS_7=CS_7_numerator/len(lines)
CS_8=CS_8_numerator/len(lines)
CS_9=CS_9_numerator/len(lines)
CS_10=CS_10_numerator/len(lines)

print(CS_1)
print(CS_2)
print(CS_3)
print(CS_4)
print(CS_5)
print(CS_6)
print(CS_7)
print(CS_8)
print(CS_9)
print(CS_10)