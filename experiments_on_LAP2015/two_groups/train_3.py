#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:40:00 2019

@author: xjc
"""



import math
import numpy as np
import fire
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from Dataset_folder import Dataset_floder as data_prepare
from ensemble_learning_resnet_two_group_Chalearn2015 import el_resnet101
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, loader, optimizer, epoch, n_epochs, device, print_freq=1):

    batch_time = AverageMeter()
    total_loss=AverageMeter()
    argmax_MAE=AverageMeter()
    epsilon_error=AverageMeter()
    accuracy=AverageMeter()

    # Model on train mode
    model.train()
    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        ####
        input_var, target_var = input.to(device), target.long().to(device)     
        ####
        
        # compute output
        output = model(input_var)
        batch_size=target.size(0)

                
        cross_entropy_term_1 = torch.nn.functional.cross_entropy(output[:,0:2], target_var[:,0])
        cross_entropy_term_2 = torch.nn.functional.cross_entropy(output[:,2:4], target_var[:,1])
        cross_entropy_term_3 = torch.nn.functional.cross_entropy(output[:,4:6], target_var[:,2])
        cross_entropy_term_4 = torch.nn.functional.cross_entropy(output[:,6:8], target_var[:,3])
        cross_entropy_term_5 = torch.nn.functional.cross_entropy(output[:,8:10], target_var[:,4])
        cross_entropy_term_6 = torch.nn.functional.cross_entropy(output[:,10:12], target_var[:,5])
        cross_entropy_term_7 = torch.nn.functional.cross_entropy(output[:,12:14], target_var[:,6])
        cross_entropy_term_8 = torch.nn.functional.cross_entropy(output[:,14:16], target_var[:,7])
        cross_entropy_term_9 = torch.nn.functional.cross_entropy(output[:,16:18], target_var[:,8])
        cross_entropy_term_10 = torch.nn.functional.cross_entropy(output[:,18:20], target_var[:,9])
        cross_entropy_term_11 = torch.nn.functional.cross_entropy(output[:,20:22], target_var[:,10])
        cross_entropy_term_12 = torch.nn.functional.cross_entropy(output[:,22:24], target_var[:,11])
        cross_entropy_term_13 = torch.nn.functional.cross_entropy(output[:,24:26], target_var[:,12])
        cross_entropy_term_14 = torch.nn.functional.cross_entropy(output[:,26:28], target_var[:,13])
        cross_entropy_term_15 = torch.nn.functional.cross_entropy(output[:,28:30], target_var[:,14])
        cross_entropy_term_16 = torch.nn.functional.cross_entropy(output[:,30:32], target_var[:,15])
        cross_entropy_term_17 = torch.nn.functional.cross_entropy(output[:,32:34], target_var[:,16])
        cross_entropy_term_18 = torch.nn.functional.cross_entropy(output[:,34:36], target_var[:,17])
        cross_entropy_term_19 = torch.nn.functional.cross_entropy(output[:,36:38], target_var[:,18])
        cross_entropy_term_20 = torch.nn.functional.cross_entropy(output[:,38:40], target_var[:,19])
        cross_entropy_term_21 = torch.nn.functional.cross_entropy(output[:,40:42], target_var[:,20])
        cross_entropy_term_22 = torch.nn.functional.cross_entropy(output[:,42:44], target_var[:,21])
        cross_entropy_term_23 = torch.nn.functional.cross_entropy(output[:,44:46], target_var[:,22])
        cross_entropy_term_24 = torch.nn.functional.cross_entropy(output[:,46:48], target_var[:,23])
        cross_entropy_term_25 = torch.nn.functional.cross_entropy(output[:,48:50], target_var[:,24])
        cross_entropy_term_26 = torch.nn.functional.cross_entropy(output[:,50:52], target_var[:,25])
        cross_entropy_term_27 = torch.nn.functional.cross_entropy(output[:,52:54], target_var[:,26])
        cross_entropy_term_28 = torch.nn.functional.cross_entropy(output[:,54:56], target_var[:,27])
        cross_entropy_term_29 = torch.nn.functional.cross_entropy(output[:,56:58], target_var[:,28])
        cross_entropy_term_30 = torch.nn.functional.cross_entropy(output[:,58:60], target_var[:,29])
        cross_entropy_term_31 = torch.nn.functional.cross_entropy(output[:,60:62], target_var[:,30])
        cross_entropy_term_32 = torch.nn.functional.cross_entropy(output[:,62:64], target_var[:,31])
        cross_entropy_term_33 = torch.nn.functional.cross_entropy(output[:,64:66], target_var[:,32])
        cross_entropy_term_34 = torch.nn.functional.cross_entropy(output[:,66:68], target_var[:,33])
        cross_entropy_term_35 = torch.nn.functional.cross_entropy(output[:,68:70], target_var[:,34])
        cross_entropy_term_36 = torch.nn.functional.cross_entropy(output[:,70:72], target_var[:,35])
        cross_entropy_term_37 = torch.nn.functional.cross_entropy(output[:,72:74], target_var[:,36])
        cross_entropy_term_38 = torch.nn.functional.cross_entropy(output[:,74:76], target_var[:,37])
        cross_entropy_term_39 = torch.nn.functional.cross_entropy(output[:,76:78], target_var[:,38])
        cross_entropy_term_40 = torch.nn.functional.cross_entropy(output[:,78:80], target_var[:,39])
        cross_entropy_term_41 = torch.nn.functional.cross_entropy(output[:,80:82], target_var[:,40])
        cross_entropy_term_42 = torch.nn.functional.cross_entropy(output[:,82:84], target_var[:,41])
        cross_entropy_term_43 = torch.nn.functional.cross_entropy(output[:,84:86], target_var[:,42])
        cross_entropy_term_44 = torch.nn.functional.cross_entropy(output[:,86:88], target_var[:,43])
        cross_entropy_term_45 = torch.nn.functional.cross_entropy(output[:,88:90], target_var[:,44])
        cross_entropy_term_46 = torch.nn.functional.cross_entropy(output[:,90:92], target_var[:,45])
        cross_entropy_term_47 = torch.nn.functional.cross_entropy(output[:,92:94], target_var[:,46])
        cross_entropy_term_48 = torch.nn.functional.cross_entropy(output[:,94:96], target_var[:,47])
        cross_entropy_term_49 = torch.nn.functional.cross_entropy(output[:,96:98], target_var[:,48])
        cross_entropy_term_50 = torch.nn.functional.cross_entropy(output[:,98:100], target_var[:,49])
        cross_entropy_term_51 = torch.nn.functional.cross_entropy(output[:,100:102], target_var[:,50])
        cross_entropy_term_52 = torch.nn.functional.cross_entropy(output[:,102:104], target_var[:,51])
        cross_entropy_term_53 = torch.nn.functional.cross_entropy(output[:,104:106], target_var[:,52])
        cross_entropy_term_54 = torch.nn.functional.cross_entropy(output[:,106:108], target_var[:,53])
        cross_entropy_term_55 = torch.nn.functional.cross_entropy(output[:,108:110], target_var[:,54])
        cross_entropy_term_56 = torch.nn.functional.cross_entropy(output[:,110:112], target_var[:,55])
        cross_entropy_term_57 = torch.nn.functional.cross_entropy(output[:,112:114], target_var[:,56])
        cross_entropy_term_58 = torch.nn.functional.cross_entropy(output[:,114:116], target_var[:,57])
        cross_entropy_term_59 = torch.nn.functional.cross_entropy(output[:,116:118], target_var[:,58])
        cross_entropy_term_60 = torch.nn.functional.cross_entropy(output[:,118:120], target_var[:,59])
        cross_entropy_term_61 = torch.nn.functional.cross_entropy(output[:,120:122], target_var[:,60])
        cross_entropy_term_62 = torch.nn.functional.cross_entropy(output[:,122:124], target_var[:,61])
        cross_entropy_term_63 = torch.nn.functional.cross_entropy(output[:,124:126], target_var[:,62])
        cross_entropy_term_64 = torch.nn.functional.cross_entropy(output[:,126:128], target_var[:,63])
        cross_entropy_term_65 = torch.nn.functional.cross_entropy(output[:,128:130], target_var[:,64])
        cross_entropy_term_66 = torch.nn.functional.cross_entropy(output[:,130:132], target_var[:,65])
        cross_entropy_term_67 = torch.nn.functional.cross_entropy(output[:,132:134], target_var[:,66])
        cross_entropy_term_68 = torch.nn.functional.cross_entropy(output[:,134:136], target_var[:,67])
        cross_entropy_term_69 = torch.nn.functional.cross_entropy(output[:,136:138], target_var[:,68])
        cross_entropy_term_70 = torch.nn.functional.cross_entropy(output[:,138:140], target_var[:,69])
        cross_entropy_term_71 = torch.nn.functional.cross_entropy(output[:,140:142], target_var[:,70])
        cross_entropy_term_72 = torch.nn.functional.cross_entropy(output[:,142:144], target_var[:,71])
        cross_entropy_term_73 = torch.nn.functional.cross_entropy(output[:,144:146], target_var[:,72])
        cross_entropy_term_74 = torch.nn.functional.cross_entropy(output[:,146:148], target_var[:,73])
        cross_entropy_term_75 = torch.nn.functional.cross_entropy(output[:,148:150], target_var[:,74])
        cross_entropy_term_76 = torch.nn.functional.cross_entropy(output[:,150:152], target_var[:,75])
        cross_entropy_term_77 = torch.nn.functional.cross_entropy(output[:,152:154], target_var[:,76])
        cross_entropy_term_78 = torch.nn.functional.cross_entropy(output[:,154:156], target_var[:,77])
        cross_entropy_term_79 = torch.nn.functional.cross_entropy(output[:,156:158], target_var[:,78])
        cross_entropy_term_80 = torch.nn.functional.cross_entropy(output[:,158:160], target_var[:,79])
        cross_entropy_term_81 = torch.nn.functional.cross_entropy(output[:,160:162], target_var[:,80])
        cross_entropy_term_82 = torch.nn.functional.cross_entropy(output[:,162:164], target_var[:,81])
        cross_entropy_term_83 = torch.nn.functional.cross_entropy(output[:,164:166], target_var[:,82])
        cross_entropy_term_84 = torch.nn.functional.cross_entropy(output[:,166:168], target_var[:,83])
        cross_entropy_term_85 = torch.nn.functional.cross_entropy(output[:,168:170], target_var[:,84])
        cross_entropy_term_86 = torch.nn.functional.cross_entropy(output[:,170:172], target_var[:,85])
        cross_entropy_term_87 = torch.nn.functional.cross_entropy(output[:,172:174], target_var[:,86])
        cross_entropy_term_88 = torch.nn.functional.cross_entropy(output[:,174:176], target_var[:,87])
        cross_entropy_term_89 = torch.nn.functional.cross_entropy(output[:,176:178], target_var[:,88])
        cross_entropy_term_90 = torch.nn.functional.cross_entropy(output[:,178:180], target_var[:,89])
        cross_entropy_term_91 = torch.nn.functional.cross_entropy(output[:,180:182], target_var[:,90])
        cross_entropy_term_92 = torch.nn.functional.cross_entropy(output[:,182:184], target_var[:,91])
        cross_entropy_term_93 = torch.nn.functional.cross_entropy(output[:,184:186], target_var[:,92])
        cross_entropy_term_94 = torch.nn.functional.cross_entropy(output[:,186:188], target_var[:,93])
        cross_entropy_term_95 = torch.nn.functional.cross_entropy(output[:,188:190], target_var[:,94])
        cross_entropy_term_96 = torch.nn.functional.cross_entropy(output[:,190:192], target_var[:,95])
        cross_entropy_term_97 = torch.nn.functional.cross_entropy(output[:,192:194], target_var[:,96])
        cross_entropy_term_98 = torch.nn.functional.cross_entropy(output[:,194:196], target_var[:,97])
        cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,196:198], target_var[:,98])
        cross_entropy_term_100 = torch.nn.functional.cross_entropy(output[:,198:200], target_var[:,99])
 

        total_cross_entropy_term=(cross_entropy_term_1+cross_entropy_term_2+cross_entropy_term_3+cross_entropy_term_4+cross_entropy_term_5+
                           cross_entropy_term_6+cross_entropy_term_7+cross_entropy_term_8+cross_entropy_term_9+cross_entropy_term_10+
                           cross_entropy_term_11+cross_entropy_term_12+cross_entropy_term_13+cross_entropy_term_14+cross_entropy_term_15+
                           cross_entropy_term_16+cross_entropy_term_17+cross_entropy_term_18+cross_entropy_term_19+cross_entropy_term_20+
                           cross_entropy_term_21+cross_entropy_term_22+cross_entropy_term_23+cross_entropy_term_24+cross_entropy_term_25+
                           cross_entropy_term_26+cross_entropy_term_27+cross_entropy_term_28+cross_entropy_term_29+cross_entropy_term_30+
                           cross_entropy_term_31+cross_entropy_term_32+cross_entropy_term_33+cross_entropy_term_34+cross_entropy_term_35+
                           cross_entropy_term_36+cross_entropy_term_37+cross_entropy_term_38+cross_entropy_term_39+cross_entropy_term_40+
                           cross_entropy_term_41+cross_entropy_term_42+cross_entropy_term_43+cross_entropy_term_44+cross_entropy_term_45+
                           cross_entropy_term_46+cross_entropy_term_47+cross_entropy_term_48+cross_entropy_term_49+cross_entropy_term_50+
                           cross_entropy_term_51+cross_entropy_term_52+cross_entropy_term_53+cross_entropy_term_54+cross_entropy_term_55+
                           cross_entropy_term_56+cross_entropy_term_57+cross_entropy_term_58+cross_entropy_term_59+cross_entropy_term_60+
                           cross_entropy_term_61+cross_entropy_term_62+cross_entropy_term_63+cross_entropy_term_64+cross_entropy_term_65+
                           cross_entropy_term_66+cross_entropy_term_67+cross_entropy_term_68+cross_entropy_term_69+cross_entropy_term_70+
                           cross_entropy_term_71+cross_entropy_term_72+cross_entropy_term_73+cross_entropy_term_74+cross_entropy_term_75+
                           cross_entropy_term_76+cross_entropy_term_77+cross_entropy_term_78+cross_entropy_term_79+cross_entropy_term_80+
                           cross_entropy_term_81+cross_entropy_term_82+cross_entropy_term_83+cross_entropy_term_84+cross_entropy_term_85+
                           cross_entropy_term_86+cross_entropy_term_87+cross_entropy_term_88+cross_entropy_term_89+cross_entropy_term_90+
                           cross_entropy_term_91+cross_entropy_term_92+cross_entropy_term_93+cross_entropy_term_94+cross_entropy_term_95+
                           cross_entropy_term_96+cross_entropy_term_97+cross_entropy_term_98+cross_entropy_term_99+cross_entropy_term_100)


        loss=total_cross_entropy_term
        batch_mae, batch_epsilon_error, batch_acc, _=argmax_mae_acc(output.cpu(),target)

        total_loss.update(loss.item(), batch_size)
        argmax_MAE.update(batch_mae.item(), batch_size)
        epsilon_error.update(batch_epsilon_error, batch_size)
        accuracy.update(batch_acc.item(),batch_size)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

#        # print stats
#        if batch_idx % print_freq == 0:
#            res = '\t'.join([
#                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
#                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                'total_loss %.4f (%.4f)' % (total_loss.val, total_loss.avg),
#                'cross_entropy_Loss %.4f (%.4f)' % (cross_entropy_loss.val, cross_entropy_loss.avg),
#                'kl_loss %.4f (%.4f)' % (kl_loss.val, kl_loss.avg),
#                'argmax_MAE %.3f (%.3f)' % (argmax_MAE.val, argmax_MAE.avg),
#                'accuracy %.3f (%.3f)' % (accuracy.val, accuracy.avg)
#            ])
#            print(res)

    # Return summary statistics
    return batch_time.avg, total_loss.avg, argmax_MAE.avg, epsilon_error.avg, accuracy.avg





def test_1(model, loader, device, model_state_dir, print_freq=1):

    batch_time = AverageMeter()
#    cross_entropy_loss=AverageMeter()  
#    kl_loss=AverageMeter()
    total_loss=AverageMeter()
    argmax_MAE=AverageMeter()
    epsilon_error=AverageMeter()
    accuracy=AverageMeter()

    # Model on train mode
    model.eval()
    AE_list=[]
    predict_age_list=[]
    real_age_list=[]

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            ######
            input_var, target_var = input.to(device), target.long().to(device)     
            ######
            
            # compute output
            model.load_state_dict(torch.load(model_state_dir))
            output = model(input_var)
    
            batch_size=target.size(0)

            cross_entropy_term_1 = torch.nn.functional.cross_entropy(output[:,0:2], target_var[:,0])
            cross_entropy_term_2 = torch.nn.functional.cross_entropy(output[:,2:4], target_var[:,1])
            cross_entropy_term_3 = torch.nn.functional.cross_entropy(output[:,4:6], target_var[:,2])
            cross_entropy_term_4 = torch.nn.functional.cross_entropy(output[:,6:8], target_var[:,3])
            cross_entropy_term_5 = torch.nn.functional.cross_entropy(output[:,8:10], target_var[:,4])
            cross_entropy_term_6 = torch.nn.functional.cross_entropy(output[:,10:12], target_var[:,5])
            cross_entropy_term_7 = torch.nn.functional.cross_entropy(output[:,12:14], target_var[:,6])
            cross_entropy_term_8 = torch.nn.functional.cross_entropy(output[:,14:16], target_var[:,7])
            cross_entropy_term_9 = torch.nn.functional.cross_entropy(output[:,16:18], target_var[:,8])
            cross_entropy_term_10 = torch.nn.functional.cross_entropy(output[:,18:20], target_var[:,9])
            cross_entropy_term_11 = torch.nn.functional.cross_entropy(output[:,20:22], target_var[:,10])
            cross_entropy_term_12 = torch.nn.functional.cross_entropy(output[:,22:24], target_var[:,11])
            cross_entropy_term_13 = torch.nn.functional.cross_entropy(output[:,24:26], target_var[:,12])
            cross_entropy_term_14 = torch.nn.functional.cross_entropy(output[:,26:28], target_var[:,13])
            cross_entropy_term_15 = torch.nn.functional.cross_entropy(output[:,28:30], target_var[:,14])
            cross_entropy_term_16 = torch.nn.functional.cross_entropy(output[:,30:32], target_var[:,15])
            cross_entropy_term_17 = torch.nn.functional.cross_entropy(output[:,32:34], target_var[:,16])
            cross_entropy_term_18 = torch.nn.functional.cross_entropy(output[:,34:36], target_var[:,17])
            cross_entropy_term_19 = torch.nn.functional.cross_entropy(output[:,36:38], target_var[:,18])
            cross_entropy_term_20 = torch.nn.functional.cross_entropy(output[:,38:40], target_var[:,19])
            cross_entropy_term_21 = torch.nn.functional.cross_entropy(output[:,40:42], target_var[:,20])
            cross_entropy_term_22 = torch.nn.functional.cross_entropy(output[:,42:44], target_var[:,21])
            cross_entropy_term_23 = torch.nn.functional.cross_entropy(output[:,44:46], target_var[:,22])
            cross_entropy_term_24 = torch.nn.functional.cross_entropy(output[:,46:48], target_var[:,23])
            cross_entropy_term_25 = torch.nn.functional.cross_entropy(output[:,48:50], target_var[:,24])
            cross_entropy_term_26 = torch.nn.functional.cross_entropy(output[:,50:52], target_var[:,25])
            cross_entropy_term_27 = torch.nn.functional.cross_entropy(output[:,52:54], target_var[:,26])
            cross_entropy_term_28 = torch.nn.functional.cross_entropy(output[:,54:56], target_var[:,27])
            cross_entropy_term_29 = torch.nn.functional.cross_entropy(output[:,56:58], target_var[:,28])
            cross_entropy_term_30 = torch.nn.functional.cross_entropy(output[:,58:60], target_var[:,29])
            cross_entropy_term_31 = torch.nn.functional.cross_entropy(output[:,60:62], target_var[:,30])
            cross_entropy_term_32 = torch.nn.functional.cross_entropy(output[:,62:64], target_var[:,31])
            cross_entropy_term_33 = torch.nn.functional.cross_entropy(output[:,64:66], target_var[:,32])
            cross_entropy_term_34 = torch.nn.functional.cross_entropy(output[:,66:68], target_var[:,33])
            cross_entropy_term_35 = torch.nn.functional.cross_entropy(output[:,68:70], target_var[:,34])
            cross_entropy_term_36 = torch.nn.functional.cross_entropy(output[:,70:72], target_var[:,35])
            cross_entropy_term_37 = torch.nn.functional.cross_entropy(output[:,72:74], target_var[:,36])
            cross_entropy_term_38 = torch.nn.functional.cross_entropy(output[:,74:76], target_var[:,37])
            cross_entropy_term_39 = torch.nn.functional.cross_entropy(output[:,76:78], target_var[:,38])
            cross_entropy_term_40 = torch.nn.functional.cross_entropy(output[:,78:80], target_var[:,39])
            cross_entropy_term_41 = torch.nn.functional.cross_entropy(output[:,80:82], target_var[:,40])
            cross_entropy_term_42 = torch.nn.functional.cross_entropy(output[:,82:84], target_var[:,41])
            cross_entropy_term_43 = torch.nn.functional.cross_entropy(output[:,84:86], target_var[:,42])
            cross_entropy_term_44 = torch.nn.functional.cross_entropy(output[:,86:88], target_var[:,43])
            cross_entropy_term_45 = torch.nn.functional.cross_entropy(output[:,88:90], target_var[:,44])
            cross_entropy_term_46 = torch.nn.functional.cross_entropy(output[:,90:92], target_var[:,45])
            cross_entropy_term_47 = torch.nn.functional.cross_entropy(output[:,92:94], target_var[:,46])
            cross_entropy_term_48 = torch.nn.functional.cross_entropy(output[:,94:96], target_var[:,47])
            cross_entropy_term_49 = torch.nn.functional.cross_entropy(output[:,96:98], target_var[:,48])
            cross_entropy_term_50 = torch.nn.functional.cross_entropy(output[:,98:100], target_var[:,49])
            cross_entropy_term_51 = torch.nn.functional.cross_entropy(output[:,100:102], target_var[:,50])
            cross_entropy_term_52 = torch.nn.functional.cross_entropy(output[:,102:104], target_var[:,51])
            cross_entropy_term_53 = torch.nn.functional.cross_entropy(output[:,104:106], target_var[:,52])
            cross_entropy_term_54 = torch.nn.functional.cross_entropy(output[:,106:108], target_var[:,53])
            cross_entropy_term_55 = torch.nn.functional.cross_entropy(output[:,108:110], target_var[:,54])
            cross_entropy_term_56 = torch.nn.functional.cross_entropy(output[:,110:112], target_var[:,55])
            cross_entropy_term_57 = torch.nn.functional.cross_entropy(output[:,112:114], target_var[:,56])
            cross_entropy_term_58 = torch.nn.functional.cross_entropy(output[:,114:116], target_var[:,57])
            cross_entropy_term_59 = torch.nn.functional.cross_entropy(output[:,116:118], target_var[:,58])
            cross_entropy_term_60 = torch.nn.functional.cross_entropy(output[:,118:120], target_var[:,59])
            cross_entropy_term_61 = torch.nn.functional.cross_entropy(output[:,120:122], target_var[:,60])
            cross_entropy_term_62 = torch.nn.functional.cross_entropy(output[:,122:124], target_var[:,61])
            cross_entropy_term_63 = torch.nn.functional.cross_entropy(output[:,124:126], target_var[:,62])
            cross_entropy_term_64 = torch.nn.functional.cross_entropy(output[:,126:128], target_var[:,63])
            cross_entropy_term_65 = torch.nn.functional.cross_entropy(output[:,128:130], target_var[:,64])
            cross_entropy_term_66 = torch.nn.functional.cross_entropy(output[:,130:132], target_var[:,65])
            cross_entropy_term_67 = torch.nn.functional.cross_entropy(output[:,132:134], target_var[:,66])
            cross_entropy_term_68 = torch.nn.functional.cross_entropy(output[:,134:136], target_var[:,67])
            cross_entropy_term_69 = torch.nn.functional.cross_entropy(output[:,136:138], target_var[:,68])
            cross_entropy_term_70 = torch.nn.functional.cross_entropy(output[:,138:140], target_var[:,69])
            cross_entropy_term_71 = torch.nn.functional.cross_entropy(output[:,140:142], target_var[:,70])
            cross_entropy_term_72 = torch.nn.functional.cross_entropy(output[:,142:144], target_var[:,71])
            cross_entropy_term_73 = torch.nn.functional.cross_entropy(output[:,144:146], target_var[:,72])
            cross_entropy_term_74 = torch.nn.functional.cross_entropy(output[:,146:148], target_var[:,73])
            cross_entropy_term_75 = torch.nn.functional.cross_entropy(output[:,148:150], target_var[:,74])
            cross_entropy_term_76 = torch.nn.functional.cross_entropy(output[:,150:152], target_var[:,75])
            cross_entropy_term_77 = torch.nn.functional.cross_entropy(output[:,152:154], target_var[:,76])
            cross_entropy_term_78 = torch.nn.functional.cross_entropy(output[:,154:156], target_var[:,77])
            cross_entropy_term_79 = torch.nn.functional.cross_entropy(output[:,156:158], target_var[:,78])
            cross_entropy_term_80 = torch.nn.functional.cross_entropy(output[:,158:160], target_var[:,79])
            cross_entropy_term_81 = torch.nn.functional.cross_entropy(output[:,160:162], target_var[:,80])
            cross_entropy_term_82 = torch.nn.functional.cross_entropy(output[:,162:164], target_var[:,81])
            cross_entropy_term_83 = torch.nn.functional.cross_entropy(output[:,164:166], target_var[:,82])
            cross_entropy_term_84 = torch.nn.functional.cross_entropy(output[:,166:168], target_var[:,83])
            cross_entropy_term_85 = torch.nn.functional.cross_entropy(output[:,168:170], target_var[:,84])
            cross_entropy_term_86 = torch.nn.functional.cross_entropy(output[:,170:172], target_var[:,85])
            cross_entropy_term_87 = torch.nn.functional.cross_entropy(output[:,172:174], target_var[:,86])
            cross_entropy_term_88 = torch.nn.functional.cross_entropy(output[:,174:176], target_var[:,87])
            cross_entropy_term_89 = torch.nn.functional.cross_entropy(output[:,176:178], target_var[:,88])
            cross_entropy_term_90 = torch.nn.functional.cross_entropy(output[:,178:180], target_var[:,89])
            cross_entropy_term_91 = torch.nn.functional.cross_entropy(output[:,180:182], target_var[:,90])
            cross_entropy_term_92 = torch.nn.functional.cross_entropy(output[:,182:184], target_var[:,91])
            cross_entropy_term_93 = torch.nn.functional.cross_entropy(output[:,184:186], target_var[:,92])
            cross_entropy_term_94 = torch.nn.functional.cross_entropy(output[:,186:188], target_var[:,93])
            cross_entropy_term_95 = torch.nn.functional.cross_entropy(output[:,188:190], target_var[:,94])
            cross_entropy_term_96 = torch.nn.functional.cross_entropy(output[:,190:192], target_var[:,95])
            cross_entropy_term_97 = torch.nn.functional.cross_entropy(output[:,192:194], target_var[:,96])
            cross_entropy_term_98 = torch.nn.functional.cross_entropy(output[:,194:196], target_var[:,97])
            cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,196:198], target_var[:,98])
            cross_entropy_term_100 = torch.nn.functional.cross_entropy(output[:,198:200], target_var[:,99])
     
    
            total_cross_entropy_term=(cross_entropy_term_1+cross_entropy_term_2+cross_entropy_term_3+cross_entropy_term_4+cross_entropy_term_5+
                               cross_entropy_term_6+cross_entropy_term_7+cross_entropy_term_8+cross_entropy_term_9+cross_entropy_term_10+
                               cross_entropy_term_11+cross_entropy_term_12+cross_entropy_term_13+cross_entropy_term_14+cross_entropy_term_15+
                               cross_entropy_term_16+cross_entropy_term_17+cross_entropy_term_18+cross_entropy_term_19+cross_entropy_term_20+
                               cross_entropy_term_21+cross_entropy_term_22+cross_entropy_term_23+cross_entropy_term_24+cross_entropy_term_25+
                               cross_entropy_term_26+cross_entropy_term_27+cross_entropy_term_28+cross_entropy_term_29+cross_entropy_term_30+
                               cross_entropy_term_31+cross_entropy_term_32+cross_entropy_term_33+cross_entropy_term_34+cross_entropy_term_35+
                               cross_entropy_term_36+cross_entropy_term_37+cross_entropy_term_38+cross_entropy_term_39+cross_entropy_term_40+
                               cross_entropy_term_41+cross_entropy_term_42+cross_entropy_term_43+cross_entropy_term_44+cross_entropy_term_45+
                               cross_entropy_term_46+cross_entropy_term_47+cross_entropy_term_48+cross_entropy_term_49+cross_entropy_term_50+
                               cross_entropy_term_51+cross_entropy_term_52+cross_entropy_term_53+cross_entropy_term_54+cross_entropy_term_55+
                               cross_entropy_term_56+cross_entropy_term_57+cross_entropy_term_58+cross_entropy_term_59+cross_entropy_term_60+
                               cross_entropy_term_61+cross_entropy_term_62+cross_entropy_term_63+cross_entropy_term_64+cross_entropy_term_65+
                               cross_entropy_term_66+cross_entropy_term_67+cross_entropy_term_68+cross_entropy_term_69+cross_entropy_term_70+
                               cross_entropy_term_71+cross_entropy_term_72+cross_entropy_term_73+cross_entropy_term_74+cross_entropy_term_75+
                               cross_entropy_term_76+cross_entropy_term_77+cross_entropy_term_78+cross_entropy_term_79+cross_entropy_term_80+
                               cross_entropy_term_81+cross_entropy_term_82+cross_entropy_term_83+cross_entropy_term_84+cross_entropy_term_85+
                               cross_entropy_term_86+cross_entropy_term_87+cross_entropy_term_88+cross_entropy_term_89+cross_entropy_term_90+
                               cross_entropy_term_91+cross_entropy_term_92+cross_entropy_term_93+cross_entropy_term_94+cross_entropy_term_95+
                               cross_entropy_term_96+cross_entropy_term_97+cross_entropy_term_98+cross_entropy_term_99+cross_entropy_term_100)
                

    
            loss=total_cross_entropy_term    
#            ARGMAX_MAE,acc=argmax_mae_acc(output.cpu(), target)
            batch_mae, batch_epsilon_error, batch_acc, batch_pred_ages=argmax_mae_acc(output.cpu(), target)
                
            predict_age_list.extend(batch_pred_ages)
            real_age_list.extend(target[:,100])
            batch_AE=torch.abs(batch_pred_ages.float()-target[:,100])
            AE_list.extend(batch_AE)

            total_loss.update(loss.item(), batch_size)
            argmax_MAE.update(batch_mae.item(), batch_size)
            epsilon_error.update(batch_epsilon_error, batch_size)
            accuracy.update(batch_acc.item(),batch_size)
    
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
#            # print stats
#            if batch_idx % print_freq == 0:
#                res = '\t'.join([
#                    'Test', 
#                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                    'total_loss %.4f (%.4f)' % (total_loss.val, total_loss.avg),
#                    'cross_entropy_Loss %.4f (%.4f)' % (cross_entropy_loss.val, cross_entropy_loss.avg),
#                    'kl_loss %.4f (%.4f)' % (kl_loss.val, kl_loss.avg),
#                    'argmax_MAE %.3f (%.3f)' % (argmax_MAE.val, argmax_MAE.avg),
#                    'accuracy %.3f (%.3f)' % (accuracy.val, accuracy.avg)
#                ])
#                print(res)
    
        # Return summary statistics
        return batch_time.avg, total_loss.avg, argmax_MAE.avg, epsilon_error.avg, accuracy.avg, AE_list, predict_age_list, real_age_list





def test_2(model, loader, device, model_state_dir, print_freq=1):

    batch_time = AverageMeter()
#    cross_entropy_loss=AverageMeter()  
#    kl_loss=AverageMeter()
    total_loss=AverageMeter()
    argmax_MAE=AverageMeter()
    epsilon_error=AverageMeter()
    accuracy=AverageMeter()

    # Model on train mode
    model.eval()
    AE_list=[]
    predict_age_list=[]
    real_age_list=[]

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            ######
#            input_var, target_var = input.to(device), target.long().to(device)
            bs, ncrops, c, h, w = input.size()
            target_var=target.long().to(device)
            ######
            
            # compute output
            model.load_state_dict(torch.load(model_state_dir))
#            output = model(input_var)
            result = model(input.view(-1, c, h, w).to(device))
            output = result.view(bs, ncrops, -1).mean(1)
    
            batch_size=target.size(0)

            cross_entropy_term_1 = torch.nn.functional.cross_entropy(output[:,0:2], target_var[:,0])
            cross_entropy_term_2 = torch.nn.functional.cross_entropy(output[:,2:4], target_var[:,1])
            cross_entropy_term_3 = torch.nn.functional.cross_entropy(output[:,4:6], target_var[:,2])
            cross_entropy_term_4 = torch.nn.functional.cross_entropy(output[:,6:8], target_var[:,3])
            cross_entropy_term_5 = torch.nn.functional.cross_entropy(output[:,8:10], target_var[:,4])
            cross_entropy_term_6 = torch.nn.functional.cross_entropy(output[:,10:12], target_var[:,5])
            cross_entropy_term_7 = torch.nn.functional.cross_entropy(output[:,12:14], target_var[:,6])
            cross_entropy_term_8 = torch.nn.functional.cross_entropy(output[:,14:16], target_var[:,7])
            cross_entropy_term_9 = torch.nn.functional.cross_entropy(output[:,16:18], target_var[:,8])
            cross_entropy_term_10 = torch.nn.functional.cross_entropy(output[:,18:20], target_var[:,9])
            cross_entropy_term_11 = torch.nn.functional.cross_entropy(output[:,20:22], target_var[:,10])
            cross_entropy_term_12 = torch.nn.functional.cross_entropy(output[:,22:24], target_var[:,11])
            cross_entropy_term_13 = torch.nn.functional.cross_entropy(output[:,24:26], target_var[:,12])
            cross_entropy_term_14 = torch.nn.functional.cross_entropy(output[:,26:28], target_var[:,13])
            cross_entropy_term_15 = torch.nn.functional.cross_entropy(output[:,28:30], target_var[:,14])
            cross_entropy_term_16 = torch.nn.functional.cross_entropy(output[:,30:32], target_var[:,15])
            cross_entropy_term_17 = torch.nn.functional.cross_entropy(output[:,32:34], target_var[:,16])
            cross_entropy_term_18 = torch.nn.functional.cross_entropy(output[:,34:36], target_var[:,17])
            cross_entropy_term_19 = torch.nn.functional.cross_entropy(output[:,36:38], target_var[:,18])
            cross_entropy_term_20 = torch.nn.functional.cross_entropy(output[:,38:40], target_var[:,19])
            cross_entropy_term_21 = torch.nn.functional.cross_entropy(output[:,40:42], target_var[:,20])
            cross_entropy_term_22 = torch.nn.functional.cross_entropy(output[:,42:44], target_var[:,21])
            cross_entropy_term_23 = torch.nn.functional.cross_entropy(output[:,44:46], target_var[:,22])
            cross_entropy_term_24 = torch.nn.functional.cross_entropy(output[:,46:48], target_var[:,23])
            cross_entropy_term_25 = torch.nn.functional.cross_entropy(output[:,48:50], target_var[:,24])
            cross_entropy_term_26 = torch.nn.functional.cross_entropy(output[:,50:52], target_var[:,25])
            cross_entropy_term_27 = torch.nn.functional.cross_entropy(output[:,52:54], target_var[:,26])
            cross_entropy_term_28 = torch.nn.functional.cross_entropy(output[:,54:56], target_var[:,27])
            cross_entropy_term_29 = torch.nn.functional.cross_entropy(output[:,56:58], target_var[:,28])
            cross_entropy_term_30 = torch.nn.functional.cross_entropy(output[:,58:60], target_var[:,29])
            cross_entropy_term_31 = torch.nn.functional.cross_entropy(output[:,60:62], target_var[:,30])
            cross_entropy_term_32 = torch.nn.functional.cross_entropy(output[:,62:64], target_var[:,31])
            cross_entropy_term_33 = torch.nn.functional.cross_entropy(output[:,64:66], target_var[:,32])
            cross_entropy_term_34 = torch.nn.functional.cross_entropy(output[:,66:68], target_var[:,33])
            cross_entropy_term_35 = torch.nn.functional.cross_entropy(output[:,68:70], target_var[:,34])
            cross_entropy_term_36 = torch.nn.functional.cross_entropy(output[:,70:72], target_var[:,35])
            cross_entropy_term_37 = torch.nn.functional.cross_entropy(output[:,72:74], target_var[:,36])
            cross_entropy_term_38 = torch.nn.functional.cross_entropy(output[:,74:76], target_var[:,37])
            cross_entropy_term_39 = torch.nn.functional.cross_entropy(output[:,76:78], target_var[:,38])
            cross_entropy_term_40 = torch.nn.functional.cross_entropy(output[:,78:80], target_var[:,39])
            cross_entropy_term_41 = torch.nn.functional.cross_entropy(output[:,80:82], target_var[:,40])
            cross_entropy_term_42 = torch.nn.functional.cross_entropy(output[:,82:84], target_var[:,41])
            cross_entropy_term_43 = torch.nn.functional.cross_entropy(output[:,84:86], target_var[:,42])
            cross_entropy_term_44 = torch.nn.functional.cross_entropy(output[:,86:88], target_var[:,43])
            cross_entropy_term_45 = torch.nn.functional.cross_entropy(output[:,88:90], target_var[:,44])
            cross_entropy_term_46 = torch.nn.functional.cross_entropy(output[:,90:92], target_var[:,45])
            cross_entropy_term_47 = torch.nn.functional.cross_entropy(output[:,92:94], target_var[:,46])
            cross_entropy_term_48 = torch.nn.functional.cross_entropy(output[:,94:96], target_var[:,47])
            cross_entropy_term_49 = torch.nn.functional.cross_entropy(output[:,96:98], target_var[:,48])
            cross_entropy_term_50 = torch.nn.functional.cross_entropy(output[:,98:100], target_var[:,49])
            cross_entropy_term_51 = torch.nn.functional.cross_entropy(output[:,100:102], target_var[:,50])
            cross_entropy_term_52 = torch.nn.functional.cross_entropy(output[:,102:104], target_var[:,51])
            cross_entropy_term_53 = torch.nn.functional.cross_entropy(output[:,104:106], target_var[:,52])
            cross_entropy_term_54 = torch.nn.functional.cross_entropy(output[:,106:108], target_var[:,53])
            cross_entropy_term_55 = torch.nn.functional.cross_entropy(output[:,108:110], target_var[:,54])
            cross_entropy_term_56 = torch.nn.functional.cross_entropy(output[:,110:112], target_var[:,55])
            cross_entropy_term_57 = torch.nn.functional.cross_entropy(output[:,112:114], target_var[:,56])
            cross_entropy_term_58 = torch.nn.functional.cross_entropy(output[:,114:116], target_var[:,57])
            cross_entropy_term_59 = torch.nn.functional.cross_entropy(output[:,116:118], target_var[:,58])
            cross_entropy_term_60 = torch.nn.functional.cross_entropy(output[:,118:120], target_var[:,59])
            cross_entropy_term_61 = torch.nn.functional.cross_entropy(output[:,120:122], target_var[:,60])
            cross_entropy_term_62 = torch.nn.functional.cross_entropy(output[:,122:124], target_var[:,61])
            cross_entropy_term_63 = torch.nn.functional.cross_entropy(output[:,124:126], target_var[:,62])
            cross_entropy_term_64 = torch.nn.functional.cross_entropy(output[:,126:128], target_var[:,63])
            cross_entropy_term_65 = torch.nn.functional.cross_entropy(output[:,128:130], target_var[:,64])
            cross_entropy_term_66 = torch.nn.functional.cross_entropy(output[:,130:132], target_var[:,65])
            cross_entropy_term_67 = torch.nn.functional.cross_entropy(output[:,132:134], target_var[:,66])
            cross_entropy_term_68 = torch.nn.functional.cross_entropy(output[:,134:136], target_var[:,67])
            cross_entropy_term_69 = torch.nn.functional.cross_entropy(output[:,136:138], target_var[:,68])
            cross_entropy_term_70 = torch.nn.functional.cross_entropy(output[:,138:140], target_var[:,69])
            cross_entropy_term_71 = torch.nn.functional.cross_entropy(output[:,140:142], target_var[:,70])
            cross_entropy_term_72 = torch.nn.functional.cross_entropy(output[:,142:144], target_var[:,71])
            cross_entropy_term_73 = torch.nn.functional.cross_entropy(output[:,144:146], target_var[:,72])
            cross_entropy_term_74 = torch.nn.functional.cross_entropy(output[:,146:148], target_var[:,73])
            cross_entropy_term_75 = torch.nn.functional.cross_entropy(output[:,148:150], target_var[:,74])
            cross_entropy_term_76 = torch.nn.functional.cross_entropy(output[:,150:152], target_var[:,75])
            cross_entropy_term_77 = torch.nn.functional.cross_entropy(output[:,152:154], target_var[:,76])
            cross_entropy_term_78 = torch.nn.functional.cross_entropy(output[:,154:156], target_var[:,77])
            cross_entropy_term_79 = torch.nn.functional.cross_entropy(output[:,156:158], target_var[:,78])
            cross_entropy_term_80 = torch.nn.functional.cross_entropy(output[:,158:160], target_var[:,79])
            cross_entropy_term_81 = torch.nn.functional.cross_entropy(output[:,160:162], target_var[:,80])
            cross_entropy_term_82 = torch.nn.functional.cross_entropy(output[:,162:164], target_var[:,81])
            cross_entropy_term_83 = torch.nn.functional.cross_entropy(output[:,164:166], target_var[:,82])
            cross_entropy_term_84 = torch.nn.functional.cross_entropy(output[:,166:168], target_var[:,83])
            cross_entropy_term_85 = torch.nn.functional.cross_entropy(output[:,168:170], target_var[:,84])
            cross_entropy_term_86 = torch.nn.functional.cross_entropy(output[:,170:172], target_var[:,85])
            cross_entropy_term_87 = torch.nn.functional.cross_entropy(output[:,172:174], target_var[:,86])
            cross_entropy_term_88 = torch.nn.functional.cross_entropy(output[:,174:176], target_var[:,87])
            cross_entropy_term_89 = torch.nn.functional.cross_entropy(output[:,176:178], target_var[:,88])
            cross_entropy_term_90 = torch.nn.functional.cross_entropy(output[:,178:180], target_var[:,89])
            cross_entropy_term_91 = torch.nn.functional.cross_entropy(output[:,180:182], target_var[:,90])
            cross_entropy_term_92 = torch.nn.functional.cross_entropy(output[:,182:184], target_var[:,91])
            cross_entropy_term_93 = torch.nn.functional.cross_entropy(output[:,184:186], target_var[:,92])
            cross_entropy_term_94 = torch.nn.functional.cross_entropy(output[:,186:188], target_var[:,93])
            cross_entropy_term_95 = torch.nn.functional.cross_entropy(output[:,188:190], target_var[:,94])
            cross_entropy_term_96 = torch.nn.functional.cross_entropy(output[:,190:192], target_var[:,95])
            cross_entropy_term_97 = torch.nn.functional.cross_entropy(output[:,192:194], target_var[:,96])
            cross_entropy_term_98 = torch.nn.functional.cross_entropy(output[:,194:196], target_var[:,97])
            cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,196:198], target_var[:,98])
            cross_entropy_term_100 = torch.nn.functional.cross_entropy(output[:,198:200], target_var[:,99])
     
    
            total_cross_entropy_term=(cross_entropy_term_1+cross_entropy_term_2+cross_entropy_term_3+cross_entropy_term_4+cross_entropy_term_5+
                               cross_entropy_term_6+cross_entropy_term_7+cross_entropy_term_8+cross_entropy_term_9+cross_entropy_term_10+
                               cross_entropy_term_11+cross_entropy_term_12+cross_entropy_term_13+cross_entropy_term_14+cross_entropy_term_15+
                               cross_entropy_term_16+cross_entropy_term_17+cross_entropy_term_18+cross_entropy_term_19+cross_entropy_term_20+
                               cross_entropy_term_21+cross_entropy_term_22+cross_entropy_term_23+cross_entropy_term_24+cross_entropy_term_25+
                               cross_entropy_term_26+cross_entropy_term_27+cross_entropy_term_28+cross_entropy_term_29+cross_entropy_term_30+
                               cross_entropy_term_31+cross_entropy_term_32+cross_entropy_term_33+cross_entropy_term_34+cross_entropy_term_35+
                               cross_entropy_term_36+cross_entropy_term_37+cross_entropy_term_38+cross_entropy_term_39+cross_entropy_term_40+
                               cross_entropy_term_41+cross_entropy_term_42+cross_entropy_term_43+cross_entropy_term_44+cross_entropy_term_45+
                               cross_entropy_term_46+cross_entropy_term_47+cross_entropy_term_48+cross_entropy_term_49+cross_entropy_term_50+
                               cross_entropy_term_51+cross_entropy_term_52+cross_entropy_term_53+cross_entropy_term_54+cross_entropy_term_55+
                               cross_entropy_term_56+cross_entropy_term_57+cross_entropy_term_58+cross_entropy_term_59+cross_entropy_term_60+
                               cross_entropy_term_61+cross_entropy_term_62+cross_entropy_term_63+cross_entropy_term_64+cross_entropy_term_65+
                               cross_entropy_term_66+cross_entropy_term_67+cross_entropy_term_68+cross_entropy_term_69+cross_entropy_term_70+
                               cross_entropy_term_71+cross_entropy_term_72+cross_entropy_term_73+cross_entropy_term_74+cross_entropy_term_75+
                               cross_entropy_term_76+cross_entropy_term_77+cross_entropy_term_78+cross_entropy_term_79+cross_entropy_term_80+
                               cross_entropy_term_81+cross_entropy_term_82+cross_entropy_term_83+cross_entropy_term_84+cross_entropy_term_85+
                               cross_entropy_term_86+cross_entropy_term_87+cross_entropy_term_88+cross_entropy_term_89+cross_entropy_term_90+
                               cross_entropy_term_91+cross_entropy_term_92+cross_entropy_term_93+cross_entropy_term_94+cross_entropy_term_95+
                               cross_entropy_term_96+cross_entropy_term_97+cross_entropy_term_98+cross_entropy_term_99+cross_entropy_term_100)
                

    
            loss=total_cross_entropy_term    
#            ARGMAX_MAE,acc=argmax_mae_acc(output.cpu(), target)
            batch_mae, batch_epsilon_error, batch_acc, batch_pred_ages=argmax_mae_acc(output.cpu(), target)
                
            predict_age_list.extend(batch_pred_ages)
            real_age_list.extend(target[:,100])
            batch_AE=torch.abs(batch_pred_ages.float()-target[:,100])
            AE_list.extend(batch_AE)

            total_loss.update(loss.item(), batch_size)
            argmax_MAE.update(batch_mae.item(), batch_size)
            epsilon_error.update(batch_epsilon_error, batch_size)
            accuracy.update(batch_acc.item(),batch_size)
    
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
#            # print stats
#            if batch_idx % print_freq == 0:
#                res = '\t'.join([
#                    'Test', 
#                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                    'total_loss %.4f (%.4f)' % (total_loss.val, total_loss.avg),
#                    'cross_entropy_Loss %.4f (%.4f)' % (cross_entropy_loss.val, cross_entropy_loss.avg),
#                    'kl_loss %.4f (%.4f)' % (kl_loss.val, kl_loss.avg),
#                    'argmax_MAE %.3f (%.3f)' % (argmax_MAE.val, argmax_MAE.avg),
#                    'accuracy %.3f (%.3f)' % (accuracy.val, accuracy.avg)
#                ])
#                print(res)
    
        # Return summary statistics
        return batch_time.avg, total_loss.avg, argmax_MAE.avg, epsilon_error.avg, accuracy.avg, AE_list, predict_age_list, real_age_list


    
    
    
def argmax_mae_acc(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        
        true_predict_count=0
        for i in range(1,101):
            predicted_group=torch.argmax(output[:,2*i-2:2*i],1)
            #######
            true_predict_count+=torch.sum(torch.eq(predicted_group,target[:,i-1].long()))
            ########
            for j in range(batch_size):
                predicted_classes=[]
                if predicted_group[j]==0:
                    predicted_g1=1
                    predicted_g2=0
                else:
                    predicted_g1=0
                    predicted_g2=1

                for k in range(0,i):
                    predicted_classes.append(predicted_g1)
                for l in range(0,101-i):
                    predicted_classes.append(predicted_g2)
                if j==0:
                    tmp_batch_predicted_classes=torch.tensor(predicted_classes).view(1,-1)
                else:
                    tmp_batch_predicted_classes=torch.cat((tmp_batch_predicted_classes, torch.tensor(predicted_classes).view(1,-1)),0)
            if i==1:
                batch_predicted_classes=tmp_batch_predicted_classes.unsqueeze(2)
            else:
                batch_predicted_classes=torch.cat((batch_predicted_classes, tmp_batch_predicted_classes.unsqueeze(2)),2) 
                  
        predicted_classes_count=torch.sum(batch_predicted_classes,2)
        predicted_ages=torch.argmax(predicted_classes_count,1)
        ######
        mae=torch.sum(torch.abs(predicted_ages.float()-target[:,100]))/batch_size
        ######
        tmp_epsilon_error=0
        for i in range(0,batch_size):
            #######
            tmp_epsilon_error+=1-math.exp(-math.pow((predicted_ages[i].float()-target[i,100]),2)/(2*math.pow(target[i,101],2)))
            #######
        epsilon_error=tmp_epsilon_error/batch_size


        
        acc=true_predict_count.float().mul_(100.0/(batch_size*100))
        
        return mae, epsilon_error, acc, predicted_ages



def demo(data_root, train_list, test_list, save, n_epochs=1,
         batch_size=64, lr=0.01, wd=0.0005, momentum=0.9, seed=None):
    """
    A demo to show off training and testing of :
    "Deep facial age estimation using conditional multitask learning with weak label esxpansion."
    Trains and evaluates a mean-variance loss on MOPPH Album2 dataset.

    Args:
        data_root (str) - path to directory where data exist
        train_list (str) - path to directory where train_data_list exist
        validation_list (str) - path to directory where validation_data_list exist
        test_list (str) - path to directory where test_data_list exist
        save (str) - path to save the model and results to 

        n_epochs (int) - number of epochs for training (default 3)
        batch_size (int) - size of minibatch (default 64)
        lr (float) - base lerning rate (default 0.001)
        wd (float) -weight deday (default 0.0001)
        momentum (float) momentum (default 0.9)
        seed (int) - manually set the random seed (default None)
    """

    

    # Mean and std value from Imagenet 
    mean=[0.485, 0.456, 0.406]
    stdv=[0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5), 
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms_1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    test_transforms_2 = transforms.Compose([
       transforms.Resize(256),
       transforms.TenCrop(224),
       transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=stdv)(transforms.ToTensor()(crop)) for crop in crops])) # returns a 4D tensor
    ])
    
    
    
    
    
    if os.path.exists(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv')):
        os.remove(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'))
    with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'w') as f:
        f.write('******************************************************************\n')
        f.write('records on Chalearn2015 dataset\n')
        f.write('******************************************************************\n')
        f.write('\n')
        f.write('\n')
    
    train_set = data_prepare(data_root=data_root, data_list=train_list, transform=train_transforms)
    test_set_1 = data_prepare(data_root=data_root, data_list=test_list, transform=test_transforms_1)    
    test_set_2 = data_prepare(data_root=data_root, data_list=test_list, transform=test_transforms_2)      

    
    ensemble_learning_model = el_resnet101(num_classes=2)    
#    pretrained_dict=model_zoo.load_url(model_urls['resnet101'])
#    model_dict=ensemble_learning_model.state_dict()
#    pretrained_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}
#    model_dict.update(pretrained_dict)
#    ensemble_learning_model.load_state_dict(model_dict)

    model=ensemble_learning_model

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Model on cuda
    use_cuda=torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare data
    if seed is not None:
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
            
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=4)
    test_loader_1 = torch.utils.data.DataLoader(test_set_1, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=4)   
    test_loader_2 = torch.utils.data.DataLoader(test_set_2, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=4)  

    # Wrap model for multi-GPUs, if necessary
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model_wrapper = model.to(device)
    model_wrapper.load_state_dict(torch.load('./el_resnet101_nesterov_2groups_imdb_wiki_model_1_1.dat'))
    
    
    
    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
#        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
#                                                         gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,60],
                                                     gamma=0.1)
    
    # Train and validate model
#    best_argmax_MAE = 100
#    model_state_name_1='el_resnet101_nesterov_2groups_model_1.dat'
#    model_state_dir_1=os.path.join(save, model_state_name_1)

    best_argmax_MAE = 100
    best_accuracy=0
    best_epsilon_error=1
    model_state_name_1='el_resnet101_nesterov_2groups_model_Chalearn_LAP2015_3_1.dat'
    model_state_dir_1=os.path.join(save, model_state_name_1)
    model_state_name_2='el_resnet101_nesterov_2groups_model_Chalearn_LAP2015_3_2.dat'
    model_state_dir_2=os.path.join(save, model_state_name_2)
    model_state_name_3='el_resnet101_nesterov_2groups_model_Chalearn_LAP2015_3_3.dat'
    model_state_dir_3=os.path.join(save, model_state_name_3)    


    
    with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
        f.write('epoch, train_total_loss, train_argmax_MAE, train_accuracy\n')

    for epoch in range(n_epochs):

        scheduler.step()
        _, train_total_loss, train_argmax_MAE, train_epsilon_error, train_accuracy = train(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            device=device
        )
#         Determine if model is the best

        if train_argmax_MAE < best_argmax_MAE:
            best_argmax_MAE = train_argmax_MAE
            if os.path.exists(model_state_dir_1):
                os.remove(model_state_dir_1)
            torch.save(model_wrapper.state_dict(), model_state_dir_1)
        if train_accuracy > best_accuracy:
            best_accuracy=train_accuracy
            if os.path.exists(model_state_dir_2):
                os.remove(model_state_dir_2)
            torch.save(model_wrapper.state_dict(), model_state_dir_2)
        if train_epsilon_error < best_epsilon_error:
            best_epsilon_error=train_epsilon_error
            if os.path.exists(model_state_dir_3):
                os.remove(model_state_dir_3)
            torch.save(model_wrapper.state_dict(), model_state_dir_3)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('%03d, %0.4f, %0.4f, %0.4f, %0.4f\n'
                    % ((epoch + 1), train_total_loss, train_argmax_MAE, train_epsilon_error, train_accuracy))
        if math.isnan(float(train_argmax_MAE)):
            break


    # Test model       
    if os.path.exists(model_state_dir_1):                   
        _, test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy, AE_list, predict_age_list, real_age_list= test_1(
            model=model_wrapper,
            loader=test_loader_1,
            device=device,
            model_state_dir=model_state_dir_1,
        )
#        os.remove(model_state_dir_1)
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy))
#            f.write('\n') 

        CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
        for i in range(len(AE_list)):
            if AE_list[i]<=1:
                CS_1_numerator+=1
            if AE_list[i]<=2:
                CS_2_numerator+=1
            if AE_list[i]<=3:
                CS_3_numerator+=1
            if AE_list[i]<=4:
                CS_4_numerator+=1
            if AE_list[i]<=5:
                CS_5_numerator+=1
            if AE_list[i]<=6:
                CS_6_numerator+=1
            if AE_list[i]<=7:
                CS_7_numerator+=1
            if AE_list[i]<=8:
                CS_8_numerator+=1
            if AE_list[i]<=9:
                CS_9_numerator+=1
            if AE_list[i]<=10:
                CS_10_numerator+=1
                
        CS_1=CS_1_numerator/len(AE_list)
        CS_2=CS_2_numerator/len(AE_list)
        CS_3=CS_3_numerator/len(AE_list)
        CS_4=CS_4_numerator/len(AE_list)
        CS_5=CS_5_numerator/len(AE_list)
        CS_6=CS_6_numerator/len(AE_list)
        CS_7=CS_7_numerator/len(AE_list)
        CS_8=CS_8_numerator/len(AE_list)
        CS_9=CS_9_numerator/len(AE_list)
        CS_10=CS_10_numerator/len(AE_list)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')






    if os.path.exists(model_state_dir_2):                   
        _, test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy, AE_list, predict_age_list, real_age_list= test_1(
            model=model_wrapper,
            loader=test_loader_1,
            device=device,
            model_state_dir=model_state_dir_2,
        )
#        os.remove(model_state_dir_2)
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy))
#            f.write('\n') 

        CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
        for i in range(len(AE_list)):
            if AE_list[i]<=1:
                CS_1_numerator+=1
            if AE_list[i]<=2:
                CS_2_numerator+=1
            if AE_list[i]<=3:
                CS_3_numerator+=1
            if AE_list[i]<=4:
                CS_4_numerator+=1
            if AE_list[i]<=5:
                CS_5_numerator+=1
            if AE_list[i]<=6:
                CS_6_numerator+=1
            if AE_list[i]<=7:
                CS_7_numerator+=1
            if AE_list[i]<=8:
                CS_8_numerator+=1
            if AE_list[i]<=9:
                CS_9_numerator+=1
            if AE_list[i]<=10:
                CS_10_numerator+=1
                
        CS_1=CS_1_numerator/len(AE_list)
        CS_2=CS_2_numerator/len(AE_list)
        CS_3=CS_3_numerator/len(AE_list)
        CS_4=CS_4_numerator/len(AE_list)
        CS_5=CS_5_numerator/len(AE_list)
        CS_6=CS_6_numerator/len(AE_list)
        CS_7=CS_7_numerator/len(AE_list)
        CS_8=CS_8_numerator/len(AE_list)
        CS_9=CS_9_numerator/len(AE_list)
        CS_10=CS_10_numerator/len(AE_list)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')








    if os.path.exists(model_state_dir_3):                   
        _, test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy, AE_list, predict_age_list, real_age_list= test_1(
            model=model_wrapper,
            loader=test_loader_1,
            device=device,
            model_state_dir=model_state_dir_3,
        )
#        os.remove(model_state_dir_3)
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy))
#            f.write('\n') 

        CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
        for i in range(len(AE_list)):
            if AE_list[i]<=1:
                CS_1_numerator+=1
            if AE_list[i]<=2:
                CS_2_numerator+=1
            if AE_list[i]<=3:
                CS_3_numerator+=1
            if AE_list[i]<=4:
                CS_4_numerator+=1
            if AE_list[i]<=5:
                CS_5_numerator+=1
            if AE_list[i]<=6:
                CS_6_numerator+=1
            if AE_list[i]<=7:
                CS_7_numerator+=1
            if AE_list[i]<=8:
                CS_8_numerator+=1
            if AE_list[i]<=9:
                CS_9_numerator+=1
            if AE_list[i]<=10:
                CS_10_numerator+=1
                
        CS_1=CS_1_numerator/len(AE_list)
        CS_2=CS_2_numerator/len(AE_list)
        CS_3=CS_3_numerator/len(AE_list)
        CS_4=CS_4_numerator/len(AE_list)
        CS_5=CS_5_numerator/len(AE_list)
        CS_6=CS_6_numerator/len(AE_list)
        CS_7=CS_7_numerator/len(AE_list)
        CS_8=CS_8_numerator/len(AE_list)
        CS_9=CS_9_numerator/len(AE_list)
        CS_10=CS_10_numerator/len(AE_list)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')




    with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
        f.write('**********************************************:\n')
        f.write('**********************************************:\n')




    if os.path.exists(model_state_dir_1):                   
        _, test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy, AE_list, predict_age_list, real_age_list= test_2(
            model=model_wrapper,
            loader=test_loader_2,
            device=device,
            model_state_dir=model_state_dir_1,
        )
#        os.remove(model_state_dir_1)
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy))
#            f.write('\n') 

        CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
        for i in range(len(AE_list)):
            if AE_list[i]<=1:
                CS_1_numerator+=1
            if AE_list[i]<=2:
                CS_2_numerator+=1
            if AE_list[i]<=3:
                CS_3_numerator+=1
            if AE_list[i]<=4:
                CS_4_numerator+=1
            if AE_list[i]<=5:
                CS_5_numerator+=1
            if AE_list[i]<=6:
                CS_6_numerator+=1
            if AE_list[i]<=7:
                CS_7_numerator+=1
            if AE_list[i]<=8:
                CS_8_numerator+=1
            if AE_list[i]<=9:
                CS_9_numerator+=1
            if AE_list[i]<=10:
                CS_10_numerator+=1
                
        CS_1=CS_1_numerator/len(AE_list)
        CS_2=CS_2_numerator/len(AE_list)
        CS_3=CS_3_numerator/len(AE_list)
        CS_4=CS_4_numerator/len(AE_list)
        CS_5=CS_5_numerator/len(AE_list)
        CS_6=CS_6_numerator/len(AE_list)
        CS_7=CS_7_numerator/len(AE_list)
        CS_8=CS_8_numerator/len(AE_list)
        CS_9=CS_9_numerator/len(AE_list)
        CS_10=CS_10_numerator/len(AE_list)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')






    if os.path.exists(model_state_dir_2):                   
        _, test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy, AE_list, predict_age_list, real_age_list= test_2(
            model=model_wrapper,
            loader=test_loader_2,
            device=device,
            model_state_dir=model_state_dir_2,
        )
#        os.remove(model_state_dir_2)
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy))
#            f.write('\n') 

        CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
        for i in range(len(AE_list)):
            if AE_list[i]<=1:
                CS_1_numerator+=1
            if AE_list[i]<=2:
                CS_2_numerator+=1
            if AE_list[i]<=3:
                CS_3_numerator+=1
            if AE_list[i]<=4:
                CS_4_numerator+=1
            if AE_list[i]<=5:
                CS_5_numerator+=1
            if AE_list[i]<=6:
                CS_6_numerator+=1
            if AE_list[i]<=7:
                CS_7_numerator+=1
            if AE_list[i]<=8:
                CS_8_numerator+=1
            if AE_list[i]<=9:
                CS_9_numerator+=1
            if AE_list[i]<=10:
                CS_10_numerator+=1
                
        CS_1=CS_1_numerator/len(AE_list)
        CS_2=CS_2_numerator/len(AE_list)
        CS_3=CS_3_numerator/len(AE_list)
        CS_4=CS_4_numerator/len(AE_list)
        CS_5=CS_5_numerator/len(AE_list)
        CS_6=CS_6_numerator/len(AE_list)
        CS_7=CS_7_numerator/len(AE_list)
        CS_8=CS_8_numerator/len(AE_list)
        CS_9=CS_9_numerator/len(AE_list)
        CS_10=CS_10_numerator/len(AE_list)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')








    if os.path.exists(model_state_dir_3):                   
        _, test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy, AE_list, predict_age_list, real_age_list= test_2(
            model=model_wrapper,
            loader=test_loader_2,
            device=device,
            model_state_dir=model_state_dir_3,
        )
#        os.remove(model_state_dir_3)
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_argmax_MAE, test_epsilon_error, test_accuracy))
#            f.write('\n') 

        CS_1_numerator=CS_2_numerator=CS_3_numerator=CS_4_numerator=CS_5_numerator=CS_6_numerator=CS_7_numerator=CS_8_numerator=CS_9_numerator=CS_10_numerator=0
        for i in range(len(AE_list)):
            if AE_list[i]<=1:
                CS_1_numerator+=1
            if AE_list[i]<=2:
                CS_2_numerator+=1
            if AE_list[i]<=3:
                CS_3_numerator+=1
            if AE_list[i]<=4:
                CS_4_numerator+=1
            if AE_list[i]<=5:
                CS_5_numerator+=1
            if AE_list[i]<=6:
                CS_6_numerator+=1
            if AE_list[i]<=7:
                CS_7_numerator+=1
            if AE_list[i]<=8:
                CS_8_numerator+=1
            if AE_list[i]<=9:
                CS_9_numerator+=1
            if AE_list[i]<=10:
                CS_10_numerator+=1
                
        CS_1=CS_1_numerator/len(AE_list)
        CS_2=CS_2_numerator/len(AE_list)
        CS_3=CS_3_numerator/len(AE_list)
        CS_4=CS_4_numerator/len(AE_list)
        CS_5=CS_5_numerator/len(AE_list)
        CS_6=CS_6_numerator/len(AE_list)
        CS_7=CS_7_numerator/len(AE_list)
        CS_8=CS_8_numerator/len(AE_list)
        CS_9=CS_9_numerator/len(AE_list)
        CS_10=CS_10_numerator/len(AE_list)
        
        with open(os.path.join(save, 'el_resnet101_2groups_nesterov_results_3.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')


            


"""
A demo to train and testv MORPH Album2 dataset with protocol S1-S2-S3.

usage:
python demo.py --data_root <path_to_data_dir> --data_list <path_to_data_list_dir> --save <path_to_save_dir>


"""
if __name__ == '__main__':
    fire.Fire(demo)