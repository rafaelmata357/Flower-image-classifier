#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  29 May 2020                                
# REVISED DATE:  20 June 2020
# PURPOSE: Use the train model to predict the class (flower type) using inference
#
#   Usage:
#    python train.py <flowers path> <checpoint> top_k <top> --category_names <category> --gpu <y/n>
#   
##

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

# Imports functions created for this program
from get_predict_args import get_predict_args
from utils import load_model
from utils import imshow
from utils import process_image
from utils import color

def sanity_check(ps, cat_to_names, idx_names):
    ''' Function for checking the model sanity, based on classroom examples.
        
        Params:
        
      
        ps           : probability classes Tensor
        cat_to_names : dictionary with the class to flower names map
        idx_names    : list with the top 5 classes index
      
        
        
        Returns : None
    '''
    
    #img = Image.open(img_path)
    class_names = []
    
    # Convert index to flower names
    for n in idx_names:
        class_names.append(cat_to_names[n])
        
    ps = ps.cpu()                          #Move tensor to CPU 
    ps = ps.data.numpy().squeeze()         #Convert torch Tensor to numpy array
    
    
 
 
    
    flower = class_names[np.argmax(ps)]    #Get the flower  name with max probability 
    print('-----'*20)
    print('Predicted Flower Name: {}'.format(color(flower)))
    print('-----'*20)
    print('Top {} Classes List {} '.format(color((len(class_names))), color(class_names)))
    print('-----'*20)
    print('Classes probabilities: {}'.format(color(ps)))
    print('-----'*20)                                       
    

    return None  

def sanity_check_image(imagepath, ps, cat_to_names, idx_names):
    ''' Function for checking the model sanity, display the image with
        the top probabilities in a chart.
        
        Params:
        
      
        ps           : probability classes Tensor
        cat_to_names : dictionary with the class to flower names map
        idx_names    : list with the top 5 classes index
      
        
        
        Returns : None
    '''
    pass

def predict(image_path, checkpoint, gpu, topk, msg=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
        Params:
        
        image_path     : dir where is the image
        model          : trainned model
        topk           : number of top classes to return
        
        Returns:
        top_p          : highest topk probabilies
        top_class_idx  : top classes index
        checkpoint.    : Trainned model checkpoint
        msg            : Bool to display feedback messages
        
    '''
    
 
    
    model, optimizer = load_model(checkpoint, gpu, msg)   #Load the saved model
    
    top_class_idx = []
    image = process_image(image_path)
    image = image.unsqueeze(0)
    idx_to_class = {value : key for (key, value) in model.class_to_idx.items()} #ReMap idx to classes
      
    #Check device available
      
    if gpu == 'y':
       device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print('Running on: ',color(device))
    
    #Move to the device 
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        logps = model.forward(image)
                           
        # Top k classifications
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        
    for i in top_class[0]:
        idx = idx_to_class[i.item()]
        top_class_idx.append(idx)
    
    return top_p, top_class_idx
    

#Main Program

if __name__ == '__main__':
    
    #Get variables from command line
    
    in_arg = get_predict_args()
    image_path = in_arg.data_dir  #'flowers/test/8/image_03319.jpg'
    filepath = in_arg.checkpoint #'checkpoint.pth'
    topk = in_arg.top_k
    cat_to_name = in_arg.category_names
    gpu = in_arg.gpu
    
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    #Get the probabilty and classes index
    prob, idx_names = predict(image_path, filepath, gpu, topk)
    
    #Check th
    sanity_check(prob, cat_to_name, idx_names)

    imshow(process_image(image_path))
    