#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  29 May 2020                                
# REVISED DATE:  31 May 2020
# PURPOSE: Define utility functions to: load the data, save the checkpoint and load the checkpoint, create Model classifier
#


# Imports python modules
from time import time, sleep
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

#New Classifier class definition, based on the model used in classroom examples

class Classifier(nn.Module):
    def __init__(self,input_size,hidden_layers,output_size,drop_p=0.2):
        
        '''Creates a feedforward NN with different input size, hidden layers and output size
        
           Parameters:
           
           input_size    : integer, size of the input layer
           hidden_layers : integer list, the sizes of the hidden layers
           output_layers : integer, size of the output layer
           drop_p        : float, the dropout probability for the network to reduce overfit
           
        '''
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        #Dropout definition
        self.dropout = nn.Dropout(p=drop_p)
    
    
    
    def forward(self, x):
        
        ''' forward method with the different functions per layer 
        
        input : features tensor
        output: logits
        
        Relu function is used for the hidden layers
        LogSoftmax for output layer  ''' 
        
        # flat input tensor if neccesary
        
        x = x.view(x.shape[0], -1)
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def load_and_transform(data_dir):
    '''
    Function to load the data and apply the transformation to 
    use in the different models.

    Parameters:
    data_dir:  images path
    
    Returns:
    train_loader : data batch for trainning the network
    val_loader   : data batch for validating the network and adjust hyper parameters
    '''

    # Path for the train and validation data
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(35),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)

    return trainloader, val_loader,testloader, train_data


def save_checkpoint(model,optimizer, input_size, output_size, drop_p, epochs, lr, train_data, filepath, arch):
    
    ''' Fucntion to saved a checkpoint for the trainned model
        Parameters :
        model      : the trainned model
        optimizer  : the optimizer used
        input_size : classifier input size
        output_size: classifier output size
        drop:p.    : dropout probability
        epochs.    : epochs used
        train_data : train_data used
        filepath.  : checkpoint filepath 

        Return     : None '''

    #Move the model to CPU mode to allow predictions without GPU resources
    model.to('cpu')
    
    model.class_to_idx = train_data.class_to_idx 
    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
              'drop_p': drop_p,    
              'classes_to_idx' : model.class_to_idx, 
              'epochs': epochs,
              'lr': lr,
              'optimizer_state_dict':optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'arch': arch}
    
    print('Start saving the model... as: {}'.format(filepath))
    torch.save(checkpoint, filepath)
    print('Model Saved!!!')

    return None


def load_model(filepath, gpu, msg=True):
    
    ''' Function to load the model in a CPU or GPU if available
        
        Parameters:
        
        filepath  : String, filepath for the checkpoint
        msg       : Boolean flag to see feedback messages about the restoring process
        
        Returns:
        model     : the pretrained model with the classifier saved
        optimizer : the optimizer and their state dict '''
     
    #Check device available
   
    if gpu == 'y':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

      
    #Get the checkpoint specified
    if msg:
        print('Start loading the saved model...')
    
   
    if str(device) == "cpu":
        
        checkpoint = torch.load(filepath,map_location="cpu") #Use CPU
    else:
        checkpoint = torch.load(filepath,map_location="cuda:0") #Use GPU
            
    # Set the pretrainned model used
    
    arch = checkpoint['arch']
    
       
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:   
        model = models.densenet121(pretrained=True)
        
    
    for param in model.parameters():
        param.requires_grad = False
    
    #Rebuild the classifier
    classifier = Classifier(input_size = checkpoint['input_size'],
                         output_size = checkpoint['output_size'],
                         hidden_layers = checkpoint['hidden_layers'],
                         drop_p = checkpoint['drop_p'])
    
    model.classifier = classifier
 
    #Load model and optimizer state dict
    model.class_to_idx = checkpoint['classes_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if msg:
        print('Model loaded!!!')
    
    return model, optimizer

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def color(field):
    ''' Function to chage the text color to cyan using ASCII Escape codes
    
        Params:
        
        field : input value to add the escape codes as string
        
        returns: str value with espace codes '''
    
    cyan  = '\x1b[96m'
    white = '\x1b[97m'
    
    return cyan + str(field) + white
        

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(image)   #Open image with PIL 
    
    #Apply Transformations using transforms
    adjust = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    imgt = adjust(img)
    
    return imgt

