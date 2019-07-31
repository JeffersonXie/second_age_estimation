#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:12:49 2019

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
    train_argmax_MAE=AverageMeter()
    train_exp_MAE=AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        input_var, target_var = input.to(device), target.to(device)     

        # compute output
        output = model(input_var)
        
        batch_size=target.size(0)
        ARGMAX_MAE=argmax_MAE(output.cpu(), target)
        EXP_MAE=exp_MAE(output.cpu(), target)

        cross_entropy_loss_term=F.cross_entropy(output, target_var[:,0].long())       
        loss=cross_entropy_loss_term


        total_loss.update(loss.item(), batch_size)
        train_argmax_MAE.update(ARGMAX_MAE.item(), batch_size)
        train_exp_MAE.update(EXP_MAE.item(), batch_size)

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
#                'Total_loss %.4f (%.4f)' % (total_loss.val, total_loss.avg),
#                'Cross_entropy_Loss %.4f (%.4f)' % (cross_entropy_loss.val, cross_entropy_loss.avg),
#                'KL_loss %.4f (%.4f)' % (kl_loss.val, kl_loss.avg),
#                'argmax_MAE %.4f (%.4f)' % (train_argmax_MAE.val, train_argmax_MAE.avg),
#                'exp_MAE %.4f (%.4f)' % (train_exp_MAE.val, train_exp_MAE.avg),
#            ])
#            print(res)

    # Return summary statistics
    return batch_time.avg, total_loss.avg, train_argmax_MAE.avg, train_exp_MAE.avg




def validate(model, loader, epoch, n_epochs, device, print_freq=1):

    batch_time = AverageMeter()
    total_loss=AverageMeter()
    validate_argmax_MAE=AverageMeter()
    validate_exp_MAE=AverageMeter()
    # Model on test mode
    model.eval()


    end = time.time()
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            input_var, target_var = input.to(device), target.to(device)     
    
            # compute output
            output = model(input_var)

            batch_size=target.size(0)
            ARGMAX_MAE=argmax_MAE(output.cpu(), target)
            EXP_MAE=exp_MAE(output.cpu(), target)            
            
            cross_entropy_loss_term=F.cross_entropy(output, target_var[:,0].long())
            
            loss=cross_entropy_loss_term
        
            total_loss.update(loss.item(),batch_size)
            validate_argmax_MAE.update(ARGMAX_MAE.item(), batch_size)
            validate_exp_MAE.update(EXP_MAE.item(), batch_size)
            
    
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#            if batch_idx % print_freq == 0:
#                res = '\t'.join([
#                    'valid',
#                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                    'Total_loss %.4f (%.4f)' % (total_loss.val, total_loss.avg),
#                    'Cross_entropy_Loss %.4f (%.4f)' % (cross_entropy_loss.val, cross_entropy_loss.avg),
#                    'KL_loss %.4f (%.4f)' % (kl_loss.val, kl_loss.avg),
#                    'argmax_MAE %.4f (%.4f)' % (validate_argmax_MAE.val, validate_argmax_MAE.avg),
#                    'exp_MAE %.4f (%.4f)' % (validate_exp_MAE.val, validate_exp_MAE.avg),
#                ])
#                print(res)
    
        # Return summary statistics
        return batch_time.avg, total_loss.avg, validate_argmax_MAE.avg, validate_exp_MAE.avg




def test(model, loader, device, model_state_dir, MAE_mode, print_freq=1): 
    
    batch_time = AverageMeter()
    total_loss=AverageMeter()
    test_MAE=AverageMeter()


    # Model on test mode
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
            
            if MAE_mode=='argmax':
                MAE=argmax_MAE(output.cpu(), target)
                
                pred_age_prob=F.softmax(output[:,0:102],1)
                batches_predict_age=torch.argmax(pred_age_prob, dim=1)  
                
                batch_AE=torch.abs(batches_predict_age.float()-target_var[:,0])
                AE_list.extend(batch_AE.cpu())
                predict_age_list.extend(batches_predict_age.cpu())
                real_age_list.extend(target[:,0])
            else:
                MAE=exp_MAE(output.cpu(), target)            
            
                age_range=np.arange(102).reshape((1,102))
                age_list=np.tile(age_range,(batch_size,1))
                age_list_var=torch.from_numpy(age_list).to(device).float()   
    
                pred_age_prob=F.softmax(output[:,0:102],1)
                batches_predict_age=torch.sum(age_list_var*pred_age_prob,dim=1)  
                
                batch_AE=torch.abs(batches_predict_age-target_var[:,0])
                AE_list.extend(batch_AE.cpu())
                predict_age_list.extend(batches_predict_age.cpu())
                real_age_list.extend(target[:,0])

            
            cross_entropy_loss_term=F.cross_entropy(output, target_var[:,0].long())
            
            loss=cross_entropy_loss_term
            
            total_loss.update(loss.item(), batch_size)
            test_MAE.update(MAE.item(), batch_size)
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    


#            if batch_idx % print_freq == 0:
#                res = '\t'.join([
#                    'test',
#                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                    'Cross_entropy_Loss %.4f (%.4f)' % (cross_entropy_loss.val, cross_entropy_loss.avg),
#                    'KL_loss %.4f (%.4f)' % (kl_loss.val, kl_loss.avg),
#                    'test_MAE %.4f (%.4f)' % (test_MAE.val, test_MAE.avg),
#                ])
#                print(res)

        # Return summary statistics
        return batch_time.avg, total_loss.avg, test_MAE.avg, AE_list, predict_age_list, real_age_list


def argmax_MAE(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)  
        pred_age_prob=F.softmax(output[:,0:102],1)
        batches_predict_age=torch.argmax(pred_age_prob, dim=1)          
        absolute_error_all=torch.sum(torch.abs(batches_predict_age.float()-target[:,0]))
        MAE=absolute_error_all/batch_size
        
        return MAE

def exp_MAE(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0) 
        age_range=np.arange(102).reshape((1,102))
        age_list=np.tile(age_range,(batch_size,1))
        age_list_var=torch.from_numpy(age_list).float()   
        pred_age_prob=F.softmax(output[:,0:102],1)
        batches_predict_age=torch.sum(age_list_var*pred_age_prob,dim=1)          
        absolute_error_all=torch.sum(torch.abs(batches_predict_age-target[:,0]))
        MAE=absolute_error_all/batch_size
        
        return MAE




def demo(data_root, train_list, validate_list, test_list, save, n_epochs=1,
         batch_size=64, lr=0.01, wd=0.0005, momentum=0.9, seed=None):
#def demo(data_root, train_list, validation_list, test_list, save, n_epochs=1,
#      batch_size=64, lr=0.001, wd=0.0005, seed=None):
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
    
    
    
    
    if os.path.exists(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv')):
        os.remove(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'))
    with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'w') as f:
        f.write('******************************************************************\n')
        f.write('records on AgeDB dataset under ''80%-20%'' protocol\n')
        f.write('******************************************************************\n')
        f.write('\n')
        f.write('\n')
    
    train_set = data_prepare(data_root=data_root, data_list=train_list, transform=train_transforms)
    valid_set = data_prepare(data_root=data_root, data_list=validate_list, transform=test_transforms)
    test_set = data_prepare(data_root=data_root, data_list=test_list, transform=test_transforms)    
    

    
    resnet18_model = models.resnet18(pretrained=True)
    fc_features=resnet18_model.fc.in_features
    resnet18_model.fc=nn.Linear(fc_features, 102)
    model=resnet18_model

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
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=4) 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=4)   

    # Wrap model for multi-GPUs, if necessary
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#        model.features = torch.nn.DataParallel(model.features)
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
    best_exp_MAE = 100
    model_state_name_1='argmax_resnet18_nesterov_train_model_1_1_1.dat'
    model_state_name_2='exp_resnet18_nesterov_train_model_1_1_2.dat'
    model_state_dir_1=os.path.join(save, model_state_name_1)
    model_state_dir_2=os.path.join(save, model_state_name_2)
    
    with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'a') as f:
#            f.write('epoch,train_loss,train_male_age_accuracy,train_female_age_accuracy,train_gender_accuracy,train_MAE\n')
        f.write('epoch, train_loss, train_argmax_MAE, train_exp_MAE, valid_loss, valid_argmax_MAE, valid_exp_MAE\n')

    for epoch in range(n_epochs):
        
        scheduler.step()
        _, train_loss, train_argmax_MAE, train_exp_MAE = train(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            device=device
        )
        _, valid_loss, valid_argmax_MAE, valid_exp_MAE = validate(
            model=model_wrapper,
            loader=valid_loader,
            epoch=epoch,
            n_epochs=n_epochs,
            device=device
        )
#         Determine if model is the best


        if valid_argmax_MAE < best_argmax_MAE:
            best_argmax_MAE = valid_argmax_MAE
            if os.path.exists(model_state_dir_1):
                os.remove(model_state_dir_1)
            torch.save(model_wrapper.state_dict(), model_state_dir_1)
        if valid_exp_MAE < best_exp_MAE:
            best_exp_MAE=valid_exp_MAE
            if os.path.exists(model_state_dir_2):
                os.remove(model_state_dir_2)
            torch.save(model_wrapper.state_dict(), model_state_dir_2)
        
        
        with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'a') as f:
            f.write('%03d, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f\n' 
                    % ((epoch + 1), train_loss, train_argmax_MAE, train_exp_MAE, valid_loss, valid_argmax_MAE, valid_exp_MAE))
#        if math.isnan(float(train_MAE)):
#            break
        if math.isnan(float(train_argmax_MAE)) or math.isnan(float(train_exp_MAE)):
            break
    
    # Test model       
    if os.path.exists(model_state_dir_1):                   
        _, test_loss, test_argmax_MAE, AE_list, predict_age_list, real_age_list= test(
            model=model_wrapper,
            loader=test_loader,
            device=device,
            MAE_mode='argmax',
            model_state_dir=model_state_dir_1
        )        
        os.remove(model_state_dir_1)
        with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'a') as f:
            f.write('test_loss, test_argmax_MAE:\n')
            f.write('%0.4f, %0.4f\n' % (test_loss, test_argmax_MAE))
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
        
        with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'a') as f:
            f.write('CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10:\n')
            f.write('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f\n'
                    % (CS_1, CS_2, CS_3, CS_4, CS_5, CS_6, CS_7, CS_8, CS_9, CS_10))
            f.write('\n')
                
#'***************************'

    if os.path.exists(model_state_dir_2):                   
        _, test_loss, test_exp_MAE, AE_list, predict_age_list, real_age_list= test(
            model=model_wrapper,
            loader=test_loader,
            device=device,
            MAE_mode='exp',
            model_state_dir=model_state_dir_2
        )              
        os.remove(model_state_dir_2)
        with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'a') as f:
            f.write('test_loss, test_exp_MAE:\n')
            f.write('%0.4f, %0.4f\n' % (test_loss, test_exp_MAE))
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
        
        with open(os.path.join(save, 'argmax_exp_cross_entropy_ResNet18_nesterov_train_1_1.csv'), 'a') as f:
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