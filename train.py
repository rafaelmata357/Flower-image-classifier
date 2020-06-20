#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  28 May 2020                                
# REVISED DATE:  31 May 2020
# PURPOSE: Train, validate and test a deep neural network to classify flower images
#          using different pretrained CNN models, save the checkpoint to reuse 4096
#          for inference.
#
#   Usage:
#    python train.py flowers --save_dir <path> --learning_rate <lr> --hidden_units <units> --epochs <epochs> 
#                            --arch <alexnet,vgg19,resnet18> --gpu <y/n>
#   
##

# Imports python modules
from time import time
import numpy as np
import json
from PIL import Image
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Imports functions created for this program
from get_train_args import get_train_args
from utils import load_and_transform
from utils import save_checkpoint
from utils import Classifier
from utils import color
#Trainning and  validation  function for the CNN Model

def train_validation(model, trainloader, val_loader, criterion, optimizer, epochs, device):
    ''' Function to train and validate the model
     
        Params:
       
        model        : model to train
        trainloader  : train image batch
        val_loader   : validate image batch
        criterion    : criterion function
        optimizer    : optimizer definition
        epochs       : epoch number
        device       : CPU or GPU when available 
        
        Returns:
        
        train_losses : list with the train losses
        val_losses   : list with the validation losses
        accuracy_list: list with the accuracy results '''
    
    running_loss = 0
    train_losses = []
    val_losses = []
    accuracy_list = []   #To graph and compare if trainning vs valuation is not overfitted

    print('Start trainning the model...with {} epochs  on device: {}\n'.format(color(epochs), color(device)))

    for epoch in range(epochs):
        batch = 0
        running_loss = 0
        for inputs, labels in trainloader:
            batch += 1
        
        # Move input and label tensors to the default device
               
            inputs, labels = inputs.to(device), labels.to(device)
        
            start = time()
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            end = time()
            diff = end - start
            
            #if batch % 10 == 0:    
                #print('Trainning Epoch: {}  Trainning Batch: {}  time: {} loss {}'.format(epoch,batch,diff,loss.item()))
        else:
            val_loss = 0
            accuracy = 0
            model.eval()   
            batch = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    batch += 1
                
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    val_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    #print('Epoch: {}  Evaluation Batch: {} loss {} '.format(epoch,batch,batch_loss.item()))   
            
            train_losses.append(running_loss/len(trainloader))
            val_losses.append(val_loss/len(val_loader))
            accuracy_list.append(accuracy/len(val_loader))
            
            
            print('Epoch: {:4} Train Loss: {:.3f} Valuation Loss: {:.3f} Accuracy: {:.3f}'.format(epoch+1,train_losses[-1],
                                                                                   val_losses[-1],accuracy_list[-1]*100))
            model.train()
    print('---'*20)
    return train_losses, val_losses, accuracy_list
    
#Test function based on classroom examples

def test(model, testloader, device):
    '''Function to test the model
       Params:
       
       model      : trainned model
       tesloader  : test images batch
       device     : device (GPU or CPU)
       
       Return:
       
       accuracy_list : a list with the accuracy for each tested batch'''

    test_loss = 0
    accuracy = 0
    accuracy_list = []
    model.eval()
    batch = 0
   
    print('Start testing the model...on device: {}\n'.format(color(device)))
    with torch.no_grad():
        for inputs, labels in testloader:
            batch += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
                # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc = torch.mean(equals.type(torch.FloatTensor)).item()
            accuracy += acc
            #print(' Batch: {} loss {} '.format(batch,batch_loss.item()))   
            accuracy_list.append(acc)
        
        print('Test loss: {:.3f} Test Accuracy: {:.3f} '.format(test_loss/len(testloader),accuracy/len(testloader)))  
        print('---'*20)
    return accuracy_list

#Main program

if __name__ == "__main__":
    
    #Get variables from command line
    in_arg = get_train_args()

    #Set variables from in_arg
    
    data_dir = in_arg.data_dir
    save_dir = in_arg.save_dir
    lr = in_arg.lr
    hidden_layers = in_arg.n
    epochs = in_arg.epochs
    arch = in_arg.arch
    gpu = in_arg.gpu 
    drop_p  = in_arg.drop_p
    
    if type(hidden_layers)== int:
        hidden_layers = [hidden_layers]

    #Load and transform the data
    trainloader, val_loader, testloader, train_dataset = load_and_transform(data_dir)
      
    #Pre-trainned network load
    
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216                        #Based on pretrained model alexnet in_features value
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088                       #Based on pretrained model vgg19 in_features value
    else:
        model = models.densenet121(pretrained=True)
        input_size = 1024                        #Based on pretrained model desenet121 in_features value
   
    #Classifier creation
 
    output_size = 102            # According with the 102 flower classes
    classifier = Classifier(input_size, hidden_layers, output_size, drop_p)
    
    print('\n')
    print('---'*20)
    print('                  Trainning CNN Parameters ')
    print('---'*20)
    print(' Data directory: {:12}        Checkpoint:    {:12}'.format(color(data_dir), color(save_dir)))
    print(' CNN Pretrained: {:12}    Learning rate: {:12} Epochs: {:12}'.format(color(arch), color(lr), color(epochs)))
    print(' Hidden Units:   {:12}    Dropout:       {:12}   GPU:    {:12}'.format(color(hidden_layers), color(drop_p), color(gpu)))
    print('---'*20)
    print(color(' Classifier'), classifier)
    print('---'*20)

    #Freeze params for the pretrainned model and avoid back propagation

    for param in model.parameters():
        param.requires_grad = False

    #Assign new Classifier to replace the one from pretrainned model

    model.classifier = classifier
   

    #Set device  CUDA when is enabled if Not CPU

    if gpu == 'y':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

  
    #Define Criterion function and optimizer

    criterion = nn.NLLLoss()  #Loss function Negative Log

    #Adam back propagation optimizer with learning rate = 0.001
   
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    #Move to the device available
    model.to(device)

    #Train and evaluate the model

    t_loss, v_loss, accuracy = train_validation(model, trainloader, val_loader, criterion, optimizer, epochs, device)
 
    #Accuracy check
    accuracy_list = test(model, testloader, device)
    
    #Save the checkpoint
    save_checkpoint(model, optimizer, input_size, output_size, drop_p, epochs, lr, train_dataset, save_dir,arch)


