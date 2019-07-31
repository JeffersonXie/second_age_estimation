#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:43:03 2019

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
from ensemble_learning_resnet_three_group_morph2 import el_resnet101
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
    cross_entropy_loss=AverageMeter()  
    kl_loss=AverageMeter()
    total_loss=AverageMeter()
    argmax_MAE=AverageMeter()
    accuracy=AverageMeter()

    # Model on train mode
    model.train()


    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        input_var, target_var = input.to(device), target.to(device)     

        
        # compute output
        output = model(input_var)

        batch_size=target.size(0)


        cross_entropy_term_1 = torch.nn.functional.cross_entropy(output[:,0:3], target_var[:,3].long())
        cross_entropy_term_2 = torch.nn.functional.cross_entropy(output[:,3:6], target_var[:,7].long())
        cross_entropy_term_3 = torch.nn.functional.cross_entropy(output[:,6:9], target_var[:,11].long())
        cross_entropy_term_4 = torch.nn.functional.cross_entropy(output[:,9:12], target_var[:,15].long())
        cross_entropy_term_5 = torch.nn.functional.cross_entropy(output[:,12:15], target_var[:,19].long())
        cross_entropy_term_6 = torch.nn.functional.cross_entropy(output[:,15:18], target_var[:,23].long())
        cross_entropy_term_7 = torch.nn.functional.cross_entropy(output[:,18:21], target_var[:,27].long())
        cross_entropy_term_8 = torch.nn.functional.cross_entropy(output[:,21:24], target_var[:,31].long())
        cross_entropy_term_9 = torch.nn.functional.cross_entropy(output[:,24:27], target_var[:,35].long())
        cross_entropy_term_10 = torch.nn.functional.cross_entropy(output[:,27:30], target_var[:,39].long())
        cross_entropy_term_11 = torch.nn.functional.cross_entropy(output[:,30:33], target_var[:,43].long())
        cross_entropy_term_12 = torch.nn.functional.cross_entropy(output[:,33:36], target_var[:,47].long())
        cross_entropy_term_13 = torch.nn.functional.cross_entropy(output[:,36:39], target_var[:,51].long())
        cross_entropy_term_14 = torch.nn.functional.cross_entropy(output[:,39:42], target_var[:,55].long())
        cross_entropy_term_15 = torch.nn.functional.cross_entropy(output[:,42:45], target_var[:,59].long())
        cross_entropy_term_16 = torch.nn.functional.cross_entropy(output[:,45:48], target_var[:,63].long())
        cross_entropy_term_17 = torch.nn.functional.cross_entropy(output[:,48:51], target_var[:,67].long())
        cross_entropy_term_18 = torch.nn.functional.cross_entropy(output[:,51:54], target_var[:,71].long())
        cross_entropy_term_19 = torch.nn.functional.cross_entropy(output[:,54:57], target_var[:,75].long())
        cross_entropy_term_20 = torch.nn.functional.cross_entropy(output[:,57:60], target_var[:,79].long())
        cross_entropy_term_21 = torch.nn.functional.cross_entropy(output[:,60:63], target_var[:,83].long())
        cross_entropy_term_22 = torch.nn.functional.cross_entropy(output[:,63:66], target_var[:,87].long())
        cross_entropy_term_23 = torch.nn.functional.cross_entropy(output[:,66:69], target_var[:,91].long())
        cross_entropy_term_24 = torch.nn.functional.cross_entropy(output[:,69:72], target_var[:,95].long())
        cross_entropy_term_25 = torch.nn.functional.cross_entropy(output[:,72:75], target_var[:,99].long())
        cross_entropy_term_26 = torch.nn.functional.cross_entropy(output[:,75:78], target_var[:,103].long())
        cross_entropy_term_27 = torch.nn.functional.cross_entropy(output[:,78:81], target_var[:,107].long())
        cross_entropy_term_28 = torch.nn.functional.cross_entropy(output[:,81:84], target_var[:,111].long())
        cross_entropy_term_29 = torch.nn.functional.cross_entropy(output[:,84:87], target_var[:,115].long())
        cross_entropy_term_30 = torch.nn.functional.cross_entropy(output[:,87:90], target_var[:,119].long())
        cross_entropy_term_31 = torch.nn.functional.cross_entropy(output[:,90:93], target_var[:,123].long())
        cross_entropy_term_32 = torch.nn.functional.cross_entropy(output[:,93:96], target_var[:,127].long())
        cross_entropy_term_33 = torch.nn.functional.cross_entropy(output[:,96:99], target_var[:,131].long())
        cross_entropy_term_34 = torch.nn.functional.cross_entropy(output[:,99:102], target_var[:,135].long())
        cross_entropy_term_35 = torch.nn.functional.cross_entropy(output[:,102:105], target_var[:,139].long())
        cross_entropy_term_36 = torch.nn.functional.cross_entropy(output[:,105:108], target_var[:,143].long())
        cross_entropy_term_37 = torch.nn.functional.cross_entropy(output[:,108:111], target_var[:,147].long())
        cross_entropy_term_38 = torch.nn.functional.cross_entropy(output[:,111:114], target_var[:,151].long())
        cross_entropy_term_39 = torch.nn.functional.cross_entropy(output[:,114:117], target_var[:,155].long())
        cross_entropy_term_40 = torch.nn.functional.cross_entropy(output[:,117:120], target_var[:,159].long())
        cross_entropy_term_41 = torch.nn.functional.cross_entropy(output[:,120:123], target_var[:,163].long())
        cross_entropy_term_42 = torch.nn.functional.cross_entropy(output[:,123:126], target_var[:,167].long())
        cross_entropy_term_43 = torch.nn.functional.cross_entropy(output[:,126:129], target_var[:,171].long())
        cross_entropy_term_44 = torch.nn.functional.cross_entropy(output[:,129:132], target_var[:,175].long())
        cross_entropy_term_45 = torch.nn.functional.cross_entropy(output[:,132:135], target_var[:,179].long())
        cross_entropy_term_46 = torch.nn.functional.cross_entropy(output[:,135:138], target_var[:,183].long())
        cross_entropy_term_47 = torch.nn.functional.cross_entropy(output[:,138:141], target_var[:,187].long())
        cross_entropy_term_48 = torch.nn.functional.cross_entropy(output[:,141:144], target_var[:,191].long())
        cross_entropy_term_49 = torch.nn.functional.cross_entropy(output[:,144:147], target_var[:,195].long())
        cross_entropy_term_50 = torch.nn.functional.cross_entropy(output[:,147:150], target_var[:,199].long())
        cross_entropy_term_51 = torch.nn.functional.cross_entropy(output[:,150:153], target_var[:,203].long())
        cross_entropy_term_52 = torch.nn.functional.cross_entropy(output[:,153:156], target_var[:,207].long())
        cross_entropy_term_53 = torch.nn.functional.cross_entropy(output[:,156:159], target_var[:,211].long())
        cross_entropy_term_54 = torch.nn.functional.cross_entropy(output[:,159:162], target_var[:,215].long())
        cross_entropy_term_55 = torch.nn.functional.cross_entropy(output[:,162:165], target_var[:,219].long())
        cross_entropy_term_56 = torch.nn.functional.cross_entropy(output[:,165:168], target_var[:,223].long())
        cross_entropy_term_57 = torch.nn.functional.cross_entropy(output[:,168:171], target_var[:,227].long())
        cross_entropy_term_58 = torch.nn.functional.cross_entropy(output[:,171:174], target_var[:,231].long())
        cross_entropy_term_59 = torch.nn.functional.cross_entropy(output[:,174:177], target_var[:,235].long())
        cross_entropy_term_60 = torch.nn.functional.cross_entropy(output[:,177:180], target_var[:,239].long())
        cross_entropy_term_61 = torch.nn.functional.cross_entropy(output[:,180:183], target_var[:,243].long())
        cross_entropy_term_62 = torch.nn.functional.cross_entropy(output[:,183:186], target_var[:,247].long())
        cross_entropy_term_63 = torch.nn.functional.cross_entropy(output[:,186:189], target_var[:,251].long())
        cross_entropy_term_64 = torch.nn.functional.cross_entropy(output[:,189:192], target_var[:,255].long())
        cross_entropy_term_65 = torch.nn.functional.cross_entropy(output[:,192:195], target_var[:,259].long())
        cross_entropy_term_66 = torch.nn.functional.cross_entropy(output[:,195:198], target_var[:,263].long())
        cross_entropy_term_67 = torch.nn.functional.cross_entropy(output[:,198:201], target_var[:,267].long())
        cross_entropy_term_68 = torch.nn.functional.cross_entropy(output[:,201:204], target_var[:,271].long())
        cross_entropy_term_69 = torch.nn.functional.cross_entropy(output[:,204:207], target_var[:,275].long())
        cross_entropy_term_70 = torch.nn.functional.cross_entropy(output[:,207:210], target_var[:,279].long())
        cross_entropy_term_71 = torch.nn.functional.cross_entropy(output[:,210:213], target_var[:,283].long())
        cross_entropy_term_72 = torch.nn.functional.cross_entropy(output[:,213:216], target_var[:,287].long())
        cross_entropy_term_73 = torch.nn.functional.cross_entropy(output[:,216:219], target_var[:,291].long())
        cross_entropy_term_74 = torch.nn.functional.cross_entropy(output[:,219:222], target_var[:,295].long())
        cross_entropy_term_75 = torch.nn.functional.cross_entropy(output[:,222:225], target_var[:,299].long())
        cross_entropy_term_76 = torch.nn.functional.cross_entropy(output[:,225:228], target_var[:,303].long())
        cross_entropy_term_77 = torch.nn.functional.cross_entropy(output[:,228:231], target_var[:,307].long())
        cross_entropy_term_78 = torch.nn.functional.cross_entropy(output[:,231:234], target_var[:,311].long())
        cross_entropy_term_79 = torch.nn.functional.cross_entropy(output[:,234:237], target_var[:,315].long())
        cross_entropy_term_80 = torch.nn.functional.cross_entropy(output[:,237:240], target_var[:,319].long())
        cross_entropy_term_81 = torch.nn.functional.cross_entropy(output[:,240:243], target_var[:,323].long())
        cross_entropy_term_82 = torch.nn.functional.cross_entropy(output[:,243:246], target_var[:,327].long())
        cross_entropy_term_83 = torch.nn.functional.cross_entropy(output[:,246:249], target_var[:,331].long())
        cross_entropy_term_84 = torch.nn.functional.cross_entropy(output[:,249:252], target_var[:,335].long())
        cross_entropy_term_85 = torch.nn.functional.cross_entropy(output[:,252:255], target_var[:,339].long())
        cross_entropy_term_86 = torch.nn.functional.cross_entropy(output[:,255:258], target_var[:,343].long())
        cross_entropy_term_87 = torch.nn.functional.cross_entropy(output[:,258:261], target_var[:,347].long())
        cross_entropy_term_88 = torch.nn.functional.cross_entropy(output[:,261:264], target_var[:,351].long())
        cross_entropy_term_89 = torch.nn.functional.cross_entropy(output[:,264:267], target_var[:,355].long())
        cross_entropy_term_90 = torch.nn.functional.cross_entropy(output[:,267:270], target_var[:,359].long())
        cross_entropy_term_91 = torch.nn.functional.cross_entropy(output[:,270:273], target_var[:,363].long())
        cross_entropy_term_92 = torch.nn.functional.cross_entropy(output[:,273:276], target_var[:,367].long())
        cross_entropy_term_93 = torch.nn.functional.cross_entropy(output[:,276:279], target_var[:,371].long())
        cross_entropy_term_94 = torch.nn.functional.cross_entropy(output[:,279:282], target_var[:,375].long())
        cross_entropy_term_95 = torch.nn.functional.cross_entropy(output[:,282:285], target_var[:,379].long())
        cross_entropy_term_96 = torch.nn.functional.cross_entropy(output[:,285:288], target_var[:,383].long())
        cross_entropy_term_97 = torch.nn.functional.cross_entropy(output[:,288:291], target_var[:,387].long())
        cross_entropy_term_98 = torch.nn.functional.cross_entropy(output[:,291:294], target_var[:,391].long())
#        cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,294:397], target_var[:,395].long())
        cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,294:297], target_var[:,395].long())

        total_cross_entroy_term=(cross_entropy_term_1+cross_entropy_term_2+cross_entropy_term_3+cross_entropy_term_4+cross_entropy_term_5+
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
                           cross_entropy_term_96+cross_entropy_term_97+cross_entropy_term_98+cross_entropy_term_99)
                           
        KLloss=nn.KLDivLoss(reduction='sum')
        kl_loss_term_1=KLloss(F.log_softmax(output[:,0:3],1), target_var[:,0:3])/batch_size
        kl_loss_term_2=KLloss(F.log_softmax(output[:,3:6],1), target_var[:,4:7])/batch_size
        kl_loss_term_3=KLloss(F.log_softmax(output[:,6:9],1), target_var[:,8:11])/batch_size
        kl_loss_term_4=KLloss(F.log_softmax(output[:,9:12],1), target_var[:,12:15])/batch_size
        kl_loss_term_5=KLloss(F.log_softmax(output[:,12:15],1), target_var[:,16:19])/batch_size
        kl_loss_term_6=KLloss(F.log_softmax(output[:,15:18],1), target_var[:,20:23])/batch_size
        kl_loss_term_7=KLloss(F.log_softmax(output[:,18:21],1), target_var[:,24:27])/batch_size
        kl_loss_term_8=KLloss(F.log_softmax(output[:,21:24],1), target_var[:,28:31])/batch_size
        kl_loss_term_9=KLloss(F.log_softmax(output[:,24:27],1), target_var[:,32:35])/batch_size
        kl_loss_term_10=KLloss(F.log_softmax(output[:,27:30],1), target_var[:,36:39])/batch_size
        kl_loss_term_11=KLloss(F.log_softmax(output[:,30:33],1), target_var[:,40:43])/batch_size
        kl_loss_term_12=KLloss(F.log_softmax(output[:,33:36],1), target_var[:,44:47])/batch_size
        kl_loss_term_13=KLloss(F.log_softmax(output[:,36:39],1), target_var[:,48:51])/batch_size
        kl_loss_term_14=KLloss(F.log_softmax(output[:,39:42],1), target_var[:,52:55])/batch_size
        kl_loss_term_15=KLloss(F.log_softmax(output[:,42:45],1), target_var[:,56:59])/batch_size
        kl_loss_term_16=KLloss(F.log_softmax(output[:,45:48],1), target_var[:,60:63])/batch_size
        kl_loss_term_17=KLloss(F.log_softmax(output[:,48:51],1), target_var[:,64:67])/batch_size
        kl_loss_term_18=KLloss(F.log_softmax(output[:,51:54],1), target_var[:,68:71])/batch_size
        kl_loss_term_19=KLloss(F.log_softmax(output[:,54:57],1), target_var[:,72:75])/batch_size
        kl_loss_term_20=KLloss(F.log_softmax(output[:,57:60],1), target_var[:,76:79])/batch_size
        kl_loss_term_21=KLloss(F.log_softmax(output[:,60:63],1), target_var[:,80:83])/batch_size
        kl_loss_term_22=KLloss(F.log_softmax(output[:,63:66],1), target_var[:,84:87])/batch_size
        kl_loss_term_23=KLloss(F.log_softmax(output[:,66:69],1), target_var[:,88:91])/batch_size
        kl_loss_term_24=KLloss(F.log_softmax(output[:,69:72],1), target_var[:,92:95])/batch_size
        kl_loss_term_25=KLloss(F.log_softmax(output[:,72:75],1), target_var[:,96:99])/batch_size
        kl_loss_term_26=KLloss(F.log_softmax(output[:,75:78],1), target_var[:,100:103])/batch_size
        kl_loss_term_27=KLloss(F.log_softmax(output[:,78:81],1), target_var[:,104:107])/batch_size
        kl_loss_term_28=KLloss(F.log_softmax(output[:,81:84],1), target_var[:,108:111])/batch_size
        kl_loss_term_29=KLloss(F.log_softmax(output[:,84:87],1), target_var[:,112:115])/batch_size
        kl_loss_term_30=KLloss(F.log_softmax(output[:,87:90],1), target_var[:,116:119])/batch_size
        kl_loss_term_31=KLloss(F.log_softmax(output[:,90:93],1), target_var[:,120:123])/batch_size
        kl_loss_term_32=KLloss(F.log_softmax(output[:,93:96],1), target_var[:,124:127])/batch_size
        kl_loss_term_33=KLloss(F.log_softmax(output[:,96:99],1), target_var[:,128:131])/batch_size
        kl_loss_term_34=KLloss(F.log_softmax(output[:,99:102],1), target_var[:,132:135])/batch_size
        kl_loss_term_35=KLloss(F.log_softmax(output[:,102:105],1), target_var[:,136:139])/batch_size
        kl_loss_term_36=KLloss(F.log_softmax(output[:,105:108],1), target_var[:,140:143])/batch_size
        kl_loss_term_37=KLloss(F.log_softmax(output[:,108:111],1), target_var[:,144:147])/batch_size
        kl_loss_term_38=KLloss(F.log_softmax(output[:,111:114],1), target_var[:,148:151])/batch_size
        kl_loss_term_39=KLloss(F.log_softmax(output[:,114:117],1), target_var[:,152:155])/batch_size
        kl_loss_term_40=KLloss(F.log_softmax(output[:,117:120],1), target_var[:,156:159])/batch_size
        kl_loss_term_41=KLloss(F.log_softmax(output[:,120:123],1), target_var[:,160:163])/batch_size
        kl_loss_term_42=KLloss(F.log_softmax(output[:,123:126],1), target_var[:,164:167])/batch_size
        kl_loss_term_43=KLloss(F.log_softmax(output[:,126:129],1), target_var[:,168:171])/batch_size
        kl_loss_term_44=KLloss(F.log_softmax(output[:,129:132],1), target_var[:,172:175])/batch_size
        kl_loss_term_45=KLloss(F.log_softmax(output[:,132:135],1), target_var[:,176:179])/batch_size
        kl_loss_term_46=KLloss(F.log_softmax(output[:,135:138],1), target_var[:,180:183])/batch_size
        kl_loss_term_47=KLloss(F.log_softmax(output[:,138:141],1), target_var[:,184:187])/batch_size
        kl_loss_term_48=KLloss(F.log_softmax(output[:,141:144],1), target_var[:,188:191])/batch_size
        kl_loss_term_49=KLloss(F.log_softmax(output[:,144:147],1), target_var[:,192:195])/batch_size
        kl_loss_term_50=KLloss(F.log_softmax(output[:,147:150],1), target_var[:,196:199])/batch_size
        kl_loss_term_51=KLloss(F.log_softmax(output[:,150:153],1), target_var[:,200:203])/batch_size
        kl_loss_term_52=KLloss(F.log_softmax(output[:,153:156],1), target_var[:,204:207])/batch_size
        kl_loss_term_53=KLloss(F.log_softmax(output[:,156:159],1), target_var[:,208:211])/batch_size
        kl_loss_term_54=KLloss(F.log_softmax(output[:,159:162],1), target_var[:,212:215])/batch_size
        kl_loss_term_55=KLloss(F.log_softmax(output[:,162:165],1), target_var[:,216:219])/batch_size
        kl_loss_term_56=KLloss(F.log_softmax(output[:,165:168],1), target_var[:,220:223])/batch_size
        kl_loss_term_57=KLloss(F.log_softmax(output[:,168:171],1), target_var[:,224:227])/batch_size
        kl_loss_term_58=KLloss(F.log_softmax(output[:,171:174],1), target_var[:,228:231])/batch_size
        kl_loss_term_59=KLloss(F.log_softmax(output[:,174:177],1), target_var[:,232:235])/batch_size
        kl_loss_term_60=KLloss(F.log_softmax(output[:,177:180],1), target_var[:,236:239])/batch_size
        kl_loss_term_61=KLloss(F.log_softmax(output[:,180:183],1), target_var[:,240:243])/batch_size
        kl_loss_term_62=KLloss(F.log_softmax(output[:,183:186],1), target_var[:,244:247])/batch_size
        kl_loss_term_63=KLloss(F.log_softmax(output[:,186:189],1), target_var[:,248:251])/batch_size
        kl_loss_term_64=KLloss(F.log_softmax(output[:,189:192],1), target_var[:,252:255])/batch_size
        kl_loss_term_65=KLloss(F.log_softmax(output[:,192:195],1), target_var[:,256:259])/batch_size
        kl_loss_term_66=KLloss(F.log_softmax(output[:,195:198],1), target_var[:,260:263])/batch_size
        kl_loss_term_67=KLloss(F.log_softmax(output[:,198:201],1), target_var[:,264:267])/batch_size
        kl_loss_term_68=KLloss(F.log_softmax(output[:,201:204],1), target_var[:,268:271])/batch_size
        kl_loss_term_69=KLloss(F.log_softmax(output[:,204:207],1), target_var[:,272:275])/batch_size
        kl_loss_term_70=KLloss(F.log_softmax(output[:,207:210],1), target_var[:,276:279])/batch_size
        kl_loss_term_71=KLloss(F.log_softmax(output[:,210:213],1), target_var[:,280:283])/batch_size
        kl_loss_term_72=KLloss(F.log_softmax(output[:,213:216],1), target_var[:,284:287])/batch_size
        kl_loss_term_73=KLloss(F.log_softmax(output[:,216:219],1), target_var[:,288:291])/batch_size
        kl_loss_term_74=KLloss(F.log_softmax(output[:,219:222],1), target_var[:,292:295])/batch_size
        kl_loss_term_75=KLloss(F.log_softmax(output[:,222:225],1), target_var[:,296:299])/batch_size
        kl_loss_term_76=KLloss(F.log_softmax(output[:,225:228],1), target_var[:,300:303])/batch_size
        kl_loss_term_77=KLloss(F.log_softmax(output[:,228:231],1), target_var[:,304:307])/batch_size
        kl_loss_term_78=KLloss(F.log_softmax(output[:,231:234],1), target_var[:,308:311])/batch_size
        kl_loss_term_79=KLloss(F.log_softmax(output[:,234:237],1), target_var[:,312:315])/batch_size
        kl_loss_term_80=KLloss(F.log_softmax(output[:,237:240],1), target_var[:,316:319])/batch_size
        kl_loss_term_81=KLloss(F.log_softmax(output[:,240:243],1), target_var[:,320:323])/batch_size
        kl_loss_term_82=KLloss(F.log_softmax(output[:,243:246],1), target_var[:,324:327])/batch_size
        kl_loss_term_83=KLloss(F.log_softmax(output[:,246:249],1), target_var[:,328:331])/batch_size
        kl_loss_term_84=KLloss(F.log_softmax(output[:,249:252],1), target_var[:,332:335])/batch_size
        kl_loss_term_85=KLloss(F.log_softmax(output[:,252:255],1), target_var[:,336:339])/batch_size
        kl_loss_term_86=KLloss(F.log_softmax(output[:,255:258],1), target_var[:,340:343])/batch_size
        kl_loss_term_87=KLloss(F.log_softmax(output[:,258:261],1), target_var[:,344:347])/batch_size
        kl_loss_term_88=KLloss(F.log_softmax(output[:,261:264],1), target_var[:,348:351])/batch_size
        kl_loss_term_89=KLloss(F.log_softmax(output[:,264:267],1), target_var[:,352:355])/batch_size
        kl_loss_term_90=KLloss(F.log_softmax(output[:,267:270],1), target_var[:,356:359])/batch_size
        kl_loss_term_91=KLloss(F.log_softmax(output[:,270:273],1), target_var[:,360:363])/batch_size
        kl_loss_term_92=KLloss(F.log_softmax(output[:,273:276],1), target_var[:,364:367])/batch_size
        kl_loss_term_93=KLloss(F.log_softmax(output[:,276:279],1), target_var[:,368:371])/batch_size
        kl_loss_term_94=KLloss(F.log_softmax(output[:,279:282],1), target_var[:,372:375])/batch_size
        kl_loss_term_95=KLloss(F.log_softmax(output[:,282:285],1), target_var[:,376:379])/batch_size
        kl_loss_term_96=KLloss(F.log_softmax(output[:,285:288],1), target_var[:,380:383])/batch_size
        kl_loss_term_97=KLloss(F.log_softmax(output[:,288:291],1), target_var[:,384:387])/batch_size
        kl_loss_term_98=KLloss(F.log_softmax(output[:,291:294],1), target_var[:,388:391])/batch_size
        kl_loss_term_99=KLloss(F.log_softmax(output[:,294:297],1), target_var[:,392:395])/batch_size
        
        total_kl_loss_term=(kl_loss_term_1+kl_loss_term_2+kl_loss_term_3+kl_loss_term_4+kl_loss_term_5+
                           kl_loss_term_6+kl_loss_term_7+kl_loss_term_8+kl_loss_term_9+kl_loss_term_10+
                           kl_loss_term_11+kl_loss_term_12+kl_loss_term_13+kl_loss_term_14+kl_loss_term_15+
                           kl_loss_term_16+kl_loss_term_17+kl_loss_term_18+kl_loss_term_19+kl_loss_term_20+
                           kl_loss_term_21+kl_loss_term_22+kl_loss_term_23+kl_loss_term_24+kl_loss_term_25+
                           kl_loss_term_26+kl_loss_term_27+kl_loss_term_28+kl_loss_term_29+kl_loss_term_30+
                           kl_loss_term_31+kl_loss_term_32+kl_loss_term_33+kl_loss_term_34+kl_loss_term_35+
                           kl_loss_term_36+kl_loss_term_37+kl_loss_term_38+kl_loss_term_39+kl_loss_term_40+
                           kl_loss_term_41+kl_loss_term_42+kl_loss_term_43+kl_loss_term_44+kl_loss_term_45+
                           kl_loss_term_46+kl_loss_term_47+kl_loss_term_48+kl_loss_term_49+kl_loss_term_50+
                           kl_loss_term_51+kl_loss_term_52+kl_loss_term_53+kl_loss_term_54+kl_loss_term_55+
                           kl_loss_term_56+kl_loss_term_57+kl_loss_term_58+kl_loss_term_59+kl_loss_term_60+
                           kl_loss_term_61+kl_loss_term_62+kl_loss_term_63+kl_loss_term_64+kl_loss_term_65+
                           kl_loss_term_66+kl_loss_term_67+kl_loss_term_68+kl_loss_term_69+kl_loss_term_70+
                           kl_loss_term_71+kl_loss_term_72+kl_loss_term_73+kl_loss_term_74+kl_loss_term_75+
                           kl_loss_term_76+kl_loss_term_77+kl_loss_term_78+kl_loss_term_79+kl_loss_term_80+
                           kl_loss_term_81+kl_loss_term_82+kl_loss_term_83+kl_loss_term_84+kl_loss_term_85+
                           kl_loss_term_86+kl_loss_term_87+kl_loss_term_88+kl_loss_term_89+kl_loss_term_90+
                           kl_loss_term_91+kl_loss_term_92+kl_loss_term_93+kl_loss_term_94+kl_loss_term_95+
                           kl_loss_term_96+kl_loss_term_97+kl_loss_term_98+kl_loss_term_99)
                                                 

#        loss=mean_cross_entroy_term+mean_kl_loss_term
        loss=total_kl_loss_term

#        ARGMAX_MAE,acc=argmax_mae_acc(output.cpu(), target)
        batch_mae,batch_acc,_=argmax_mae_acc(output.cpu(),target)

        cross_entropy_loss.update(total_cross_entroy_term.item(), batch_size)  
        kl_loss.update(total_kl_loss_term.item(), batch_size)
        total_loss.update(loss.item(), batch_size)
        argmax_MAE.update(batch_mae.item(), batch_size)
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
    return batch_time.avg, total_loss.avg, cross_entropy_loss.avg, kl_loss.avg, argmax_MAE.avg, accuracy.avg




def test(model, loader, device, model_state_dir, print_freq=1):

    batch_time = AverageMeter()
    cross_entropy_loss=AverageMeter()  
    kl_loss=AverageMeter()
    total_loss=AverageMeter()
    argmax_MAE=AverageMeter()
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
            input_var, target_var = input.to(device), target.to(device)     
    
            
            # compute output
            model.load_state_dict(torch.load(model_state_dir))
            output = model(input_var)
    
            batch_size=target.size(0)
    
    
            cross_entropy_term_1 = torch.nn.functional.cross_entropy(output[:,0:3], target_var[:,3].long())
            cross_entropy_term_2 = torch.nn.functional.cross_entropy(output[:,3:6], target_var[:,7].long())
            cross_entropy_term_3 = torch.nn.functional.cross_entropy(output[:,6:9], target_var[:,11].long())
            cross_entropy_term_4 = torch.nn.functional.cross_entropy(output[:,9:12], target_var[:,15].long())
            cross_entropy_term_5 = torch.nn.functional.cross_entropy(output[:,12:15], target_var[:,19].long())
            cross_entropy_term_6 = torch.nn.functional.cross_entropy(output[:,15:18], target_var[:,23].long())
            cross_entropy_term_7 = torch.nn.functional.cross_entropy(output[:,18:21], target_var[:,27].long())
            cross_entropy_term_8 = torch.nn.functional.cross_entropy(output[:,21:24], target_var[:,31].long())
            cross_entropy_term_9 = torch.nn.functional.cross_entropy(output[:,24:27], target_var[:,35].long())
            cross_entropy_term_10 = torch.nn.functional.cross_entropy(output[:,27:30], target_var[:,39].long())
            cross_entropy_term_11 = torch.nn.functional.cross_entropy(output[:,30:33], target_var[:,43].long())
            cross_entropy_term_12 = torch.nn.functional.cross_entropy(output[:,33:36], target_var[:,47].long())
            cross_entropy_term_13 = torch.nn.functional.cross_entropy(output[:,36:39], target_var[:,51].long())
            cross_entropy_term_14 = torch.nn.functional.cross_entropy(output[:,39:42], target_var[:,55].long())
            cross_entropy_term_15 = torch.nn.functional.cross_entropy(output[:,42:45], target_var[:,59].long())
            cross_entropy_term_16 = torch.nn.functional.cross_entropy(output[:,45:48], target_var[:,63].long())
            cross_entropy_term_17 = torch.nn.functional.cross_entropy(output[:,48:51], target_var[:,67].long())
            cross_entropy_term_18 = torch.nn.functional.cross_entropy(output[:,51:54], target_var[:,71].long())
            cross_entropy_term_19 = torch.nn.functional.cross_entropy(output[:,54:57], target_var[:,75].long())
            cross_entropy_term_20 = torch.nn.functional.cross_entropy(output[:,57:60], target_var[:,79].long())
            cross_entropy_term_21 = torch.nn.functional.cross_entropy(output[:,60:63], target_var[:,83].long())
            cross_entropy_term_22 = torch.nn.functional.cross_entropy(output[:,63:66], target_var[:,87].long())
            cross_entropy_term_23 = torch.nn.functional.cross_entropy(output[:,66:69], target_var[:,91].long())
            cross_entropy_term_24 = torch.nn.functional.cross_entropy(output[:,69:72], target_var[:,95].long())
            cross_entropy_term_25 = torch.nn.functional.cross_entropy(output[:,72:75], target_var[:,99].long())
            cross_entropy_term_26 = torch.nn.functional.cross_entropy(output[:,75:78], target_var[:,103].long())
            cross_entropy_term_27 = torch.nn.functional.cross_entropy(output[:,78:81], target_var[:,107].long())
            cross_entropy_term_28 = torch.nn.functional.cross_entropy(output[:,81:84], target_var[:,111].long())
            cross_entropy_term_29 = torch.nn.functional.cross_entropy(output[:,84:87], target_var[:,115].long())
            cross_entropy_term_30 = torch.nn.functional.cross_entropy(output[:,87:90], target_var[:,119].long())
            cross_entropy_term_31 = torch.nn.functional.cross_entropy(output[:,90:93], target_var[:,123].long())
            cross_entropy_term_32 = torch.nn.functional.cross_entropy(output[:,93:96], target_var[:,127].long())
            cross_entropy_term_33 = torch.nn.functional.cross_entropy(output[:,96:99], target_var[:,131].long())
            cross_entropy_term_34 = torch.nn.functional.cross_entropy(output[:,99:102], target_var[:,135].long())
            cross_entropy_term_35 = torch.nn.functional.cross_entropy(output[:,102:105], target_var[:,139].long())
            cross_entropy_term_36 = torch.nn.functional.cross_entropy(output[:,105:108], target_var[:,143].long())
            cross_entropy_term_37 = torch.nn.functional.cross_entropy(output[:,108:111], target_var[:,147].long())
            cross_entropy_term_38 = torch.nn.functional.cross_entropy(output[:,111:114], target_var[:,151].long())
            cross_entropy_term_39 = torch.nn.functional.cross_entropy(output[:,114:117], target_var[:,155].long())
            cross_entropy_term_40 = torch.nn.functional.cross_entropy(output[:,117:120], target_var[:,159].long())
            cross_entropy_term_41 = torch.nn.functional.cross_entropy(output[:,120:123], target_var[:,163].long())
            cross_entropy_term_42 = torch.nn.functional.cross_entropy(output[:,123:126], target_var[:,167].long())
            cross_entropy_term_43 = torch.nn.functional.cross_entropy(output[:,126:129], target_var[:,171].long())
            cross_entropy_term_44 = torch.nn.functional.cross_entropy(output[:,129:132], target_var[:,175].long())
            cross_entropy_term_45 = torch.nn.functional.cross_entropy(output[:,132:135], target_var[:,179].long())
            cross_entropy_term_46 = torch.nn.functional.cross_entropy(output[:,135:138], target_var[:,183].long())
            cross_entropy_term_47 = torch.nn.functional.cross_entropy(output[:,138:141], target_var[:,187].long())
            cross_entropy_term_48 = torch.nn.functional.cross_entropy(output[:,141:144], target_var[:,191].long())
            cross_entropy_term_49 = torch.nn.functional.cross_entropy(output[:,144:147], target_var[:,195].long())
            cross_entropy_term_50 = torch.nn.functional.cross_entropy(output[:,147:150], target_var[:,199].long())
            cross_entropy_term_51 = torch.nn.functional.cross_entropy(output[:,150:153], target_var[:,203].long())
            cross_entropy_term_52 = torch.nn.functional.cross_entropy(output[:,153:156], target_var[:,207].long())
            cross_entropy_term_53 = torch.nn.functional.cross_entropy(output[:,156:159], target_var[:,211].long())
            cross_entropy_term_54 = torch.nn.functional.cross_entropy(output[:,159:162], target_var[:,215].long())
            cross_entropy_term_55 = torch.nn.functional.cross_entropy(output[:,162:165], target_var[:,219].long())
            cross_entropy_term_56 = torch.nn.functional.cross_entropy(output[:,165:168], target_var[:,223].long())
            cross_entropy_term_57 = torch.nn.functional.cross_entropy(output[:,168:171], target_var[:,227].long())
            cross_entropy_term_58 = torch.nn.functional.cross_entropy(output[:,171:174], target_var[:,231].long())
            cross_entropy_term_59 = torch.nn.functional.cross_entropy(output[:,174:177], target_var[:,235].long())
            cross_entropy_term_60 = torch.nn.functional.cross_entropy(output[:,177:180], target_var[:,239].long())
            cross_entropy_term_61 = torch.nn.functional.cross_entropy(output[:,180:183], target_var[:,243].long())
            cross_entropy_term_62 = torch.nn.functional.cross_entropy(output[:,183:186], target_var[:,247].long())
            cross_entropy_term_63 = torch.nn.functional.cross_entropy(output[:,186:189], target_var[:,251].long())
            cross_entropy_term_64 = torch.nn.functional.cross_entropy(output[:,189:192], target_var[:,255].long())
            cross_entropy_term_65 = torch.nn.functional.cross_entropy(output[:,192:195], target_var[:,259].long())
            cross_entropy_term_66 = torch.nn.functional.cross_entropy(output[:,195:198], target_var[:,263].long())
            cross_entropy_term_67 = torch.nn.functional.cross_entropy(output[:,198:201], target_var[:,267].long())
            cross_entropy_term_68 = torch.nn.functional.cross_entropy(output[:,201:204], target_var[:,271].long())
            cross_entropy_term_69 = torch.nn.functional.cross_entropy(output[:,204:207], target_var[:,275].long())
            cross_entropy_term_70 = torch.nn.functional.cross_entropy(output[:,207:210], target_var[:,279].long())
            cross_entropy_term_71 = torch.nn.functional.cross_entropy(output[:,210:213], target_var[:,283].long())
            cross_entropy_term_72 = torch.nn.functional.cross_entropy(output[:,213:216], target_var[:,287].long())
            cross_entropy_term_73 = torch.nn.functional.cross_entropy(output[:,216:219], target_var[:,291].long())
            cross_entropy_term_74 = torch.nn.functional.cross_entropy(output[:,219:222], target_var[:,295].long())
            cross_entropy_term_75 = torch.nn.functional.cross_entropy(output[:,222:225], target_var[:,299].long())
            cross_entropy_term_76 = torch.nn.functional.cross_entropy(output[:,225:228], target_var[:,303].long())
            cross_entropy_term_77 = torch.nn.functional.cross_entropy(output[:,228:231], target_var[:,307].long())
            cross_entropy_term_78 = torch.nn.functional.cross_entropy(output[:,231:234], target_var[:,311].long())
            cross_entropy_term_79 = torch.nn.functional.cross_entropy(output[:,234:237], target_var[:,315].long())
            cross_entropy_term_80 = torch.nn.functional.cross_entropy(output[:,237:240], target_var[:,319].long())
            cross_entropy_term_81 = torch.nn.functional.cross_entropy(output[:,240:243], target_var[:,323].long())
            cross_entropy_term_82 = torch.nn.functional.cross_entropy(output[:,243:246], target_var[:,327].long())
            cross_entropy_term_83 = torch.nn.functional.cross_entropy(output[:,246:249], target_var[:,331].long())
            cross_entropy_term_84 = torch.nn.functional.cross_entropy(output[:,249:252], target_var[:,335].long())
            cross_entropy_term_85 = torch.nn.functional.cross_entropy(output[:,252:255], target_var[:,339].long())
            cross_entropy_term_86 = torch.nn.functional.cross_entropy(output[:,255:258], target_var[:,343].long())
            cross_entropy_term_87 = torch.nn.functional.cross_entropy(output[:,258:261], target_var[:,347].long())
            cross_entropy_term_88 = torch.nn.functional.cross_entropy(output[:,261:264], target_var[:,351].long())
            cross_entropy_term_89 = torch.nn.functional.cross_entropy(output[:,264:267], target_var[:,355].long())
            cross_entropy_term_90 = torch.nn.functional.cross_entropy(output[:,267:270], target_var[:,359].long())
            cross_entropy_term_91 = torch.nn.functional.cross_entropy(output[:,270:273], target_var[:,363].long())
            cross_entropy_term_92 = torch.nn.functional.cross_entropy(output[:,273:276], target_var[:,367].long())
            cross_entropy_term_93 = torch.nn.functional.cross_entropy(output[:,276:279], target_var[:,371].long())
            cross_entropy_term_94 = torch.nn.functional.cross_entropy(output[:,279:282], target_var[:,375].long())
            cross_entropy_term_95 = torch.nn.functional.cross_entropy(output[:,282:285], target_var[:,379].long())
            cross_entropy_term_96 = torch.nn.functional.cross_entropy(output[:,285:288], target_var[:,383].long())
            cross_entropy_term_97 = torch.nn.functional.cross_entropy(output[:,288:291], target_var[:,387].long())
            cross_entropy_term_98 = torch.nn.functional.cross_entropy(output[:,291:294], target_var[:,391].long())
#            cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,294:397], target_var[:,395].long())
            cross_entropy_term_99 = torch.nn.functional.cross_entropy(output[:,294:297], target_var[:,395].long())
    
            total_cross_entroy_term=(cross_entropy_term_1+cross_entropy_term_2+cross_entropy_term_3+cross_entropy_term_4+cross_entropy_term_5+
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
                               cross_entropy_term_96+cross_entropy_term_97+cross_entropy_term_98+cross_entropy_term_99)
                               
            KLloss=nn.KLDivLoss(reduction='sum')
            kl_loss_term_1=KLloss(F.log_softmax(output[:,0:3],1), target_var[:,0:3])/batch_size
            kl_loss_term_2=KLloss(F.log_softmax(output[:,3:6],1), target_var[:,4:7])/batch_size
            kl_loss_term_3=KLloss(F.log_softmax(output[:,6:9],1), target_var[:,8:11])/batch_size
            kl_loss_term_4=KLloss(F.log_softmax(output[:,9:12],1), target_var[:,12:15])/batch_size
            kl_loss_term_5=KLloss(F.log_softmax(output[:,12:15],1), target_var[:,16:19])/batch_size
            kl_loss_term_6=KLloss(F.log_softmax(output[:,15:18],1), target_var[:,20:23])/batch_size
            kl_loss_term_7=KLloss(F.log_softmax(output[:,18:21],1), target_var[:,24:27])/batch_size
            kl_loss_term_8=KLloss(F.log_softmax(output[:,21:24],1), target_var[:,28:31])/batch_size
            kl_loss_term_9=KLloss(F.log_softmax(output[:,24:27],1), target_var[:,32:35])/batch_size
            kl_loss_term_10=KLloss(F.log_softmax(output[:,27:30],1), target_var[:,36:39])/batch_size
            kl_loss_term_11=KLloss(F.log_softmax(output[:,30:33],1), target_var[:,40:43])/batch_size
            kl_loss_term_12=KLloss(F.log_softmax(output[:,33:36],1), target_var[:,44:47])/batch_size
            kl_loss_term_13=KLloss(F.log_softmax(output[:,36:39],1), target_var[:,48:51])/batch_size
            kl_loss_term_14=KLloss(F.log_softmax(output[:,39:42],1), target_var[:,52:55])/batch_size
            kl_loss_term_15=KLloss(F.log_softmax(output[:,42:45],1), target_var[:,56:59])/batch_size
            kl_loss_term_16=KLloss(F.log_softmax(output[:,45:48],1), target_var[:,60:63])/batch_size
            kl_loss_term_17=KLloss(F.log_softmax(output[:,48:51],1), target_var[:,64:67])/batch_size
            kl_loss_term_18=KLloss(F.log_softmax(output[:,51:54],1), target_var[:,68:71])/batch_size
            kl_loss_term_19=KLloss(F.log_softmax(output[:,54:57],1), target_var[:,72:75])/batch_size
            kl_loss_term_20=KLloss(F.log_softmax(output[:,57:60],1), target_var[:,76:79])/batch_size
            kl_loss_term_21=KLloss(F.log_softmax(output[:,60:63],1), target_var[:,80:83])/batch_size
            kl_loss_term_22=KLloss(F.log_softmax(output[:,63:66],1), target_var[:,84:87])/batch_size
            kl_loss_term_23=KLloss(F.log_softmax(output[:,66:69],1), target_var[:,88:91])/batch_size
            kl_loss_term_24=KLloss(F.log_softmax(output[:,69:72],1), target_var[:,92:95])/batch_size
            kl_loss_term_25=KLloss(F.log_softmax(output[:,72:75],1), target_var[:,96:99])/batch_size
            kl_loss_term_26=KLloss(F.log_softmax(output[:,75:78],1), target_var[:,100:103])/batch_size
            kl_loss_term_27=KLloss(F.log_softmax(output[:,78:81],1), target_var[:,104:107])/batch_size
            kl_loss_term_28=KLloss(F.log_softmax(output[:,81:84],1), target_var[:,108:111])/batch_size
            kl_loss_term_29=KLloss(F.log_softmax(output[:,84:87],1), target_var[:,112:115])/batch_size
            kl_loss_term_30=KLloss(F.log_softmax(output[:,87:90],1), target_var[:,116:119])/batch_size
            kl_loss_term_31=KLloss(F.log_softmax(output[:,90:93],1), target_var[:,120:123])/batch_size
            kl_loss_term_32=KLloss(F.log_softmax(output[:,93:96],1), target_var[:,124:127])/batch_size
            kl_loss_term_33=KLloss(F.log_softmax(output[:,96:99],1), target_var[:,128:131])/batch_size
            kl_loss_term_34=KLloss(F.log_softmax(output[:,99:102],1), target_var[:,132:135])/batch_size
            kl_loss_term_35=KLloss(F.log_softmax(output[:,102:105],1), target_var[:,136:139])/batch_size
            kl_loss_term_36=KLloss(F.log_softmax(output[:,105:108],1), target_var[:,140:143])/batch_size
            kl_loss_term_37=KLloss(F.log_softmax(output[:,108:111],1), target_var[:,144:147])/batch_size
            kl_loss_term_38=KLloss(F.log_softmax(output[:,111:114],1), target_var[:,148:151])/batch_size
            kl_loss_term_39=KLloss(F.log_softmax(output[:,114:117],1), target_var[:,152:155])/batch_size
            kl_loss_term_40=KLloss(F.log_softmax(output[:,117:120],1), target_var[:,156:159])/batch_size
            kl_loss_term_41=KLloss(F.log_softmax(output[:,120:123],1), target_var[:,160:163])/batch_size
            kl_loss_term_42=KLloss(F.log_softmax(output[:,123:126],1), target_var[:,164:167])/batch_size
            kl_loss_term_43=KLloss(F.log_softmax(output[:,126:129],1), target_var[:,168:171])/batch_size
            kl_loss_term_44=KLloss(F.log_softmax(output[:,129:132],1), target_var[:,172:175])/batch_size
            kl_loss_term_45=KLloss(F.log_softmax(output[:,132:135],1), target_var[:,176:179])/batch_size
            kl_loss_term_46=KLloss(F.log_softmax(output[:,135:138],1), target_var[:,180:183])/batch_size
            kl_loss_term_47=KLloss(F.log_softmax(output[:,138:141],1), target_var[:,184:187])/batch_size
            kl_loss_term_48=KLloss(F.log_softmax(output[:,141:144],1), target_var[:,188:191])/batch_size
            kl_loss_term_49=KLloss(F.log_softmax(output[:,144:147],1), target_var[:,192:195])/batch_size
            kl_loss_term_50=KLloss(F.log_softmax(output[:,147:150],1), target_var[:,196:199])/batch_size
            kl_loss_term_51=KLloss(F.log_softmax(output[:,150:153],1), target_var[:,200:203])/batch_size
            kl_loss_term_52=KLloss(F.log_softmax(output[:,153:156],1), target_var[:,204:207])/batch_size
            kl_loss_term_53=KLloss(F.log_softmax(output[:,156:159],1), target_var[:,208:211])/batch_size
            kl_loss_term_54=KLloss(F.log_softmax(output[:,159:162],1), target_var[:,212:215])/batch_size
            kl_loss_term_55=KLloss(F.log_softmax(output[:,162:165],1), target_var[:,216:219])/batch_size
            kl_loss_term_56=KLloss(F.log_softmax(output[:,165:168],1), target_var[:,220:223])/batch_size
            kl_loss_term_57=KLloss(F.log_softmax(output[:,168:171],1), target_var[:,224:227])/batch_size
            kl_loss_term_58=KLloss(F.log_softmax(output[:,171:174],1), target_var[:,228:231])/batch_size
            kl_loss_term_59=KLloss(F.log_softmax(output[:,174:177],1), target_var[:,232:235])/batch_size
            kl_loss_term_60=KLloss(F.log_softmax(output[:,177:180],1), target_var[:,236:239])/batch_size
            kl_loss_term_61=KLloss(F.log_softmax(output[:,180:183],1), target_var[:,240:243])/batch_size
            kl_loss_term_62=KLloss(F.log_softmax(output[:,183:186],1), target_var[:,244:247])/batch_size
            kl_loss_term_63=KLloss(F.log_softmax(output[:,186:189],1), target_var[:,248:251])/batch_size
            kl_loss_term_64=KLloss(F.log_softmax(output[:,189:192],1), target_var[:,252:255])/batch_size
            kl_loss_term_65=KLloss(F.log_softmax(output[:,192:195],1), target_var[:,256:259])/batch_size
            kl_loss_term_66=KLloss(F.log_softmax(output[:,195:198],1), target_var[:,260:263])/batch_size
            kl_loss_term_67=KLloss(F.log_softmax(output[:,198:201],1), target_var[:,264:267])/batch_size
            kl_loss_term_68=KLloss(F.log_softmax(output[:,201:204],1), target_var[:,268:271])/batch_size
            kl_loss_term_69=KLloss(F.log_softmax(output[:,204:207],1), target_var[:,272:275])/batch_size
            kl_loss_term_70=KLloss(F.log_softmax(output[:,207:210],1), target_var[:,276:279])/batch_size
            kl_loss_term_71=KLloss(F.log_softmax(output[:,210:213],1), target_var[:,280:283])/batch_size
            kl_loss_term_72=KLloss(F.log_softmax(output[:,213:216],1), target_var[:,284:287])/batch_size
            kl_loss_term_73=KLloss(F.log_softmax(output[:,216:219],1), target_var[:,288:291])/batch_size
            kl_loss_term_74=KLloss(F.log_softmax(output[:,219:222],1), target_var[:,292:295])/batch_size
            kl_loss_term_75=KLloss(F.log_softmax(output[:,222:225],1), target_var[:,296:299])/batch_size
            kl_loss_term_76=KLloss(F.log_softmax(output[:,225:228],1), target_var[:,300:303])/batch_size
            kl_loss_term_77=KLloss(F.log_softmax(output[:,228:231],1), target_var[:,304:307])/batch_size
            kl_loss_term_78=KLloss(F.log_softmax(output[:,231:234],1), target_var[:,308:311])/batch_size
            kl_loss_term_79=KLloss(F.log_softmax(output[:,234:237],1), target_var[:,312:315])/batch_size
            kl_loss_term_80=KLloss(F.log_softmax(output[:,237:240],1), target_var[:,316:319])/batch_size
            kl_loss_term_81=KLloss(F.log_softmax(output[:,240:243],1), target_var[:,320:323])/batch_size
            kl_loss_term_82=KLloss(F.log_softmax(output[:,243:246],1), target_var[:,324:327])/batch_size
            kl_loss_term_83=KLloss(F.log_softmax(output[:,246:249],1), target_var[:,328:331])/batch_size
            kl_loss_term_84=KLloss(F.log_softmax(output[:,249:252],1), target_var[:,332:335])/batch_size
            kl_loss_term_85=KLloss(F.log_softmax(output[:,252:255],1), target_var[:,336:339])/batch_size
            kl_loss_term_86=KLloss(F.log_softmax(output[:,255:258],1), target_var[:,340:343])/batch_size
            kl_loss_term_87=KLloss(F.log_softmax(output[:,258:261],1), target_var[:,344:347])/batch_size
            kl_loss_term_88=KLloss(F.log_softmax(output[:,261:264],1), target_var[:,348:351])/batch_size
            kl_loss_term_89=KLloss(F.log_softmax(output[:,264:267],1), target_var[:,352:355])/batch_size
            kl_loss_term_90=KLloss(F.log_softmax(output[:,267:270],1), target_var[:,356:359])/batch_size
            kl_loss_term_91=KLloss(F.log_softmax(output[:,270:273],1), target_var[:,360:363])/batch_size
            kl_loss_term_92=KLloss(F.log_softmax(output[:,273:276],1), target_var[:,364:367])/batch_size
            kl_loss_term_93=KLloss(F.log_softmax(output[:,276:279],1), target_var[:,368:371])/batch_size
            kl_loss_term_94=KLloss(F.log_softmax(output[:,279:282],1), target_var[:,372:375])/batch_size
            kl_loss_term_95=KLloss(F.log_softmax(output[:,282:285],1), target_var[:,376:379])/batch_size
            kl_loss_term_96=KLloss(F.log_softmax(output[:,285:288],1), target_var[:,380:383])/batch_size
            kl_loss_term_97=KLloss(F.log_softmax(output[:,288:291],1), target_var[:,384:387])/batch_size
            kl_loss_term_98=KLloss(F.log_softmax(output[:,291:294],1), target_var[:,388:391])/batch_size
            kl_loss_term_99=KLloss(F.log_softmax(output[:,294:297],1), target_var[:,392:395])/batch_size
            
            total_kl_loss_term=(kl_loss_term_1+kl_loss_term_2+kl_loss_term_3+kl_loss_term_4+kl_loss_term_5+
                               kl_loss_term_6+kl_loss_term_7+kl_loss_term_8+kl_loss_term_9+kl_loss_term_10+
                               kl_loss_term_11+kl_loss_term_12+kl_loss_term_13+kl_loss_term_14+kl_loss_term_15+
                               kl_loss_term_16+kl_loss_term_17+kl_loss_term_18+kl_loss_term_19+kl_loss_term_20+
                               kl_loss_term_21+kl_loss_term_22+kl_loss_term_23+kl_loss_term_24+kl_loss_term_25+
                               kl_loss_term_26+kl_loss_term_27+kl_loss_term_28+kl_loss_term_29+kl_loss_term_30+
                               kl_loss_term_31+kl_loss_term_32+kl_loss_term_33+kl_loss_term_34+kl_loss_term_35+
                               kl_loss_term_36+kl_loss_term_37+kl_loss_term_38+kl_loss_term_39+kl_loss_term_40+
                               kl_loss_term_41+kl_loss_term_42+kl_loss_term_43+kl_loss_term_44+kl_loss_term_45+
                               kl_loss_term_46+kl_loss_term_47+kl_loss_term_48+kl_loss_term_49+kl_loss_term_50+
                               kl_loss_term_51+kl_loss_term_52+kl_loss_term_53+kl_loss_term_54+kl_loss_term_55+
                               kl_loss_term_56+kl_loss_term_57+kl_loss_term_58+kl_loss_term_59+kl_loss_term_60+
                               kl_loss_term_61+kl_loss_term_62+kl_loss_term_63+kl_loss_term_64+kl_loss_term_65+
                               kl_loss_term_66+kl_loss_term_67+kl_loss_term_68+kl_loss_term_69+kl_loss_term_70+
                               kl_loss_term_71+kl_loss_term_72+kl_loss_term_73+kl_loss_term_74+kl_loss_term_75+
                               kl_loss_term_76+kl_loss_term_77+kl_loss_term_78+kl_loss_term_79+kl_loss_term_80+
                               kl_loss_term_81+kl_loss_term_82+kl_loss_term_83+kl_loss_term_84+kl_loss_term_85+
                               kl_loss_term_86+kl_loss_term_87+kl_loss_term_88+kl_loss_term_89+kl_loss_term_90+
                               kl_loss_term_91+kl_loss_term_92+kl_loss_term_93+kl_loss_term_94+kl_loss_term_95+
                               kl_loss_term_96+kl_loss_term_97+kl_loss_term_98+kl_loss_term_99)
                                                     
    
#            loss=mean_cross_entroy_term+mean_kl_loss_term
            loss=total_kl_loss_term
    
#            ARGMAX_MAE,acc=argmax_mae_acc(output.cpu(), target)
            batch_mae,batch_acc,batch_pred_ages=argmax_mae_acc(output.cpu(), target)
    

            
            predict_age_list.extend(batch_pred_ages)
            real_age_list.extend(target[:,396].long())
            batch_AE=torch.abs(batch_pred_ages-target[:,396].long())
            AE_list.extend(batch_AE)

    
            cross_entropy_loss.update(total_cross_entroy_term.item(), batch_size)  
            kl_loss.update(total_kl_loss_term.item(), batch_size)
            total_loss.update(loss.item(), batch_size)
            argmax_MAE.update(batch_mae.item(), batch_size)
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
        return batch_time.avg, total_loss.avg, cross_entropy_loss.avg, kl_loss.avg, argmax_MAE.avg, accuracy.avg, AE_list, predict_age_list, real_age_list




def argmax_mae_acc(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        
        true_predict_count=0
        for i in range(1,100):
#            predicted_group=torch.argmax(F.softmax(output[:,3*i-3:3*i],1),1)
            predicted_group=torch.argmax(output[:,3*i-3:3*i],1)
#            print(predicted_group)
#            os._exit(0)
            true_predict_count+=torch.sum(torch.eq(predicted_group,target[:,4*i-1].long()))
            
            for j in range(batch_size):
                predicted_classes=[]
                if predicted_group[j]==0:
                    predicted_g1=1
                    predicted_g2=0
                    predicted_g3=0
                elif predicted_group[j]==1:
                    predicted_g1=0
                    predicted_g2=1
                    predicted_g3=0
                else:
                    predicted_g1=0
                    predicted_g2=0
                    predicted_g3=1
                for k in range(0,i):
                    predicted_classes.append(predicted_g1)
                predicted_classes.append(predicted_g2)
                for l in range(0,100-i):
                    predicted_classes.append(predicted_g3)
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
        mae=torch.sum(torch.abs(predicted_ages-target[:,396].long())).float()/batch_size
        
        acc=true_predict_count.float().mul_(100.0/(batch_size*99))
        
        return mae, acc, predicted_ages



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
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    
    
    
    if os.path.exists(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv')):
        os.remove(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv'))
    with open(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv'), 'w') as f:
        f.write('******************************************************************\n')
        f.write('records on MORPH2 dataset under ''S1-S2-S3'' protocol\n')
        f.write('******************************************************************\n')
        f.write('\n')
        f.write('\n')
    
    train_set = data_prepare(data_root=data_root, data_list=train_list, transform=train_transforms)
    test_set = data_prepare(data_root=data_root, data_list=test_list, transform=test_transforms)    

    
    ensemble_learning_model = el_resnet101(num_classes=3)
    pretrained_dict=model_zoo.load_url(model_urls['resnet101'])
    model_dict=ensemble_learning_model.state_dict()
    pretrained_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    ensemble_learning_model.load_state_dict(model_dict)

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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=4)   

    # Wrap model for multi-GPUs, if necessary
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = model = torch.nn.DataParallel(model)
    model_wrapper = model.to(device)
    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
#        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
#                                                         gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,60],
                                                     gamma=0.1)
    
    # Train and validate model
    best_argmax_MAE = 100
    model_state_name_1='el_resnet101_nesterov_train_model_2.dat'
    model_state_dir_1=os.path.join(save, model_state_name_1)

    
    with open(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv'), 'a') as f:
        f.write('epoch, train_total_loss, train_cross_entropy_loss, train_kl_loss, train_argmax_MAE, train_accuracy\n')

    for epoch in range(n_epochs):

        scheduler.step()
        _, train_total_loss, train_cross_entropy_loss, train_kl_loss, train_argmax_MAE, train_accuracy = train(
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
        
        with open(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv'), 'a') as f:
            f.write('%03d, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n'
                    % ((epoch + 1), train_total_loss, train_cross_entropy_loss, train_kl_loss, train_argmax_MAE, train_accuracy))
        if math.isnan(float(train_argmax_MAE)):
            break


    # Test model       
    if os.path.exists(model_state_dir_1):                   
        _, test_total_loss, test_cross_entropy_loss, test_kl_loss, test_argmax_MAE, test_accuracy, AE_list, predict_age_list, real_age_list= test(
            model=model_wrapper,
            loader=test_loader,
            device=device,
            model_state_dir=model_state_dir_1,
        )
#        os.remove(model_state_dir_1)
        with open(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv'), 'a') as f:
            f.write('test_total_loss, test_cross_entropy_loss, test_kl_loss, test_argmax_MAE, test_accuracy:\n')
            f.write('%0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n' % (test_total_loss, test_cross_entropy_loss, test_kl_loss, test_argmax_MAE, test_accuracy))
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
        
        with open(os.path.join(save, 'el_resnet101_nesterov_results_train_2.csv'), 'a') as f:
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