
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:46:58 2019

@author: xjc
"""

import torch.nn as nn
#import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'el_resnet18', 'el_resnet34', 'el_resnet50', 'el_resnet101',
           'el_resnet152']


#model_urls = {
#    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier1 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier2 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier3 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier4 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier5 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier6 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier7 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier8 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier9 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier10 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier11 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier12 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier13 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier14 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier15 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier16 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier17 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier18 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier19 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier20 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier21 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier22 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier23 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier24 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier25 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier26 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier27 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier28 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier29 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier30 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier31 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier32 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier33 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier34 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier35 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier36 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier37 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier38 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier39 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier40 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier41 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier42 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier43 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier44 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier45 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier46 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier47 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier48 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier49 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier50 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier51 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier52 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier53 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier54 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier55 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier56 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier57 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier58 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier59 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier60 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier61 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier62 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier63 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier64 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier65 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier66 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier67 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier68 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier69 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier70 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier71 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier72 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier73 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier74 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier75 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier76 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier77 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier78 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier79 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier80 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier81 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier82 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier83 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier84 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier85 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier86 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier87 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier88 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier89 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier90 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier91 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier92 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier93 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier94 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier95 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier96 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier97 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier98 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier99 = nn.Linear(512 * block.expansion, num_classes)
        self.classifier100 = nn.Linear(512 * block.expansion, num_classes)
#

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        x4 = self.classifier4(x)
        x5 = self.classifier5(x)
        x6 = self.classifier6(x)
        x7 = self.classifier7(x)
        x8 = self.classifier8(x)
        x9 = self.classifier9(x)
        x10 = self.classifier10(x)
        x11 = self.classifier11(x)
        x12 = self.classifier12(x)
        x13 = self.classifier13(x)
        x14 = self.classifier14(x)
        x15 = self.classifier15(x)
        x16 = self.classifier16(x)
        x17 = self.classifier17(x)
        x18 = self.classifier18(x)
        x19 = self.classifier19(x)
        x20 = self.classifier20(x)
        x21 = self.classifier21(x)
        x22 = self.classifier22(x)
        x23 = self.classifier23(x)
        x24 = self.classifier24(x)
        x25 = self.classifier25(x)
        x26 = self.classifier26(x)
        x27 = self.classifier27(x)
        x28 = self.classifier28(x)
        x29 = self.classifier29(x)
        x30 = self.classifier30(x)
        x31 = self.classifier31(x)
        x32 = self.classifier32(x)
        x33 = self.classifier33(x)
        x34 = self.classifier34(x)
        x35 = self.classifier35(x)
        x36 = self.classifier36(x)
        x37 = self.classifier37(x)
        x38 = self.classifier38(x)
        x39 = self.classifier39(x)
        x40 = self.classifier40(x)
        x41 = self.classifier41(x)
        x42 = self.classifier42(x)
        x43 = self.classifier43(x)
        x44 = self.classifier44(x)
        x45 = self.classifier45(x)
        x46 = self.classifier46(x)
        x47 = self.classifier47(x)
        x48 = self.classifier48(x)
        x49 = self.classifier49(x)
        x50 = self.classifier50(x)
        x51 = self.classifier51(x)
        x52 = self.classifier52(x)
        x53 = self.classifier53(x)
        x54 = self.classifier54(x)
        x55 = self.classifier55(x)
        x56 = self.classifier56(x)
        x57 = self.classifier57(x)
        x58 = self.classifier58(x)
        x59 = self.classifier59(x)
        x60 = self.classifier60(x)
        x61 = self.classifier61(x)
        x62 = self.classifier62(x)
        x63 = self.classifier63(x)
        x64 = self.classifier64(x)
        x65 = self.classifier65(x)
        x66 = self.classifier66(x)
        x67 = self.classifier67(x)
        x68 = self.classifier68(x)
        x69 = self.classifier69(x)
        x70 = self.classifier70(x)
        x71 = self.classifier71(x)
        x72 = self.classifier72(x)
        x73 = self.classifier73(x)
        x74 = self.classifier74(x)
        x75 = self.classifier75(x)
        x76 = self.classifier76(x)
        x77 = self.classifier77(x)
        x78 = self.classifier78(x)
        x79 = self.classifier79(x)
        x80 = self.classifier80(x)
        x81 = self.classifier81(x)
        x82 = self.classifier82(x)
        x83 = self.classifier83(x)
        x84 = self.classifier84(x)
        x85 = self.classifier85(x)
        x86 = self.classifier86(x)
        x87 = self.classifier87(x)
        x88 = self.classifier88(x)
        x89 = self.classifier89(x)
        x90 = self.classifier90(x)
        x91 = self.classifier91(x)
        x92 = self.classifier92(x)
        x93 = self.classifier93(x)
        x94 = self.classifier94(x)
        x95 = self.classifier95(x)
        x96 = self.classifier96(x)
        x97 = self.classifier97(x)
        x98 = self.classifier98(x)
        x99 = self.classifier99(x)
        x100 = self.classifier100(x)

        
        out=torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,
             x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,
             x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50,x51,x52,x53,x54,x55,x56,x57,x58,x59,
             x60,x61,x62,x63,x64,x65,x66,x67,x68,x69,x70,x71,x72,x73,x74,x75,x76,x77,x78,x79,
             x80,x81,x82,x83,x84,x85,x86,x87,x88,x89,x90,x91,x92,x93,x94,x95,x96,x97,x98,x99,x100), 1)    
    
        return out


def el_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def el_resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def el_resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def el_resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def el_resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model