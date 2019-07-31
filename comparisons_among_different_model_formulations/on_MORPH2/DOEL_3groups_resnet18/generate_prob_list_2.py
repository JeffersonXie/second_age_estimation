#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:10:56 2019

@author: xjc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:04:06 2019

@author: xjc
"""


import os
import math

orig_S2_train_file='/home/xjc/test/ensemble_learning_age_estimation/S2_train_processed.txt'
orig_S2_valid_file='/home/xjc/test/ensemble_learning_age_estimation/S2_validation_processed.txt'
orig_S2_test_file='/home/xjc/test/ensemble_learning_age_estimation/S2_test_fixed_processed.txt'

new_S2_train_file='./S2_train_probs_1.txt'
new_S2_valid_file='./S2_valid_probs_1.txt'
new_S2_test_file='./S2_test_fixed_probs_1.txt'

if os.path.exists(new_S2_train_file):
    os.remove(new_S2_train_file)
if os.path.exists(new_S2_valid_file):
    os.remove(new_S2_valid_file)
if os.path.exists(new_S2_test_file):
    os.remove(new_S2_test_file)


sigma=2


#####for train lsit#########################
with open(orig_S2_train_file) as f:
    lines=f.readlines()
for line in lines:
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
    with open(new_S2_train_file, 'a') as ff:
        ff.write(tmp_record)



####for valid list ########
with open(orig_S2_valid_file) as f:
    lines=f.readlines()
for line in lines:
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
    with open(new_S2_valid_file, 'a') as ff:
        ff.write(tmp_record)
        
#######for test list#################
with open(orig_S2_test_file) as f:
    lines=f.readlines()
for line in lines:
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
    with open(new_S2_test_file, 'a') as ff:
        ff.write(tmp_record)