#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  28 May  2020                                 
# REVISED DATE :  30 May  2020
# PURPOSE: Create a function that retrieves the args for the train script
#          from the user using the Argparse Python module. If the user does not 
#          input the params default value is used. (Based on clasroom first project)
# 
# Command Line Arguments:
# 
#  1. Data Folder for train/val/ test images as --data_dir with default value 'flowers'
#  2. Save Dir to save the checkpoint as        --save_dir with default value current directory
#  3. Learning rate to used for the optimizer   --learning_rate with default value 0.001
#  4. Hidden Units to used in the classifier    --hidden_units with default value 512 (one hidden layer)
#  5. Epochs number of epocs to used trainning  --epochs with default value 5
#  6. CNN Model Architecture to used as         --arch with default value 'vgg19'
#  7. GPU to specified gpu resources use        --gpu with default value 'y'
##

# Imports python modules

import os
import argparse


# 
def get_train_args():
    '''
    Retrieves and parses the 7 command line arguments provided by the user from 
    command line. argparse module is used. 
    If some arguments is missing default value is used. 
    Command Line Arguments:
    
    Args:
     1. Data Folder for train/val/ test images as --data_dir with default value 'flowers'
     2. Save Dir to save the checkpoint as        --save_dir with default value current directory
     3. Learning rate to used for the optimizer   --learning_rate with default value 0.001
     4. Dropout probability                       --drop_p
     5. Hidden Units to used in the classifier    --hidden_units with default value 512 (one hidden layer)
     6. Epochs number of epocs to used trainning  --epochs with default value 5
     7. CNN Model Architecture to used as         --arch with default value 'vgg19'
     8. GPU to specified gpu resources use        --gpu with default value 'y'

    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    '''
    
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser('python train.py',description='Train a CNN Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
       
    # Argument 1: data path folder for train, validation and testing 
    parser.add_argument('data_dir', type = str, default= 'flower',
                    help = 'path to the images folder')  
    
    # Argument 2: checkpoint save directory
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                    help = 'path to save the checkpoint')    
    
    # Argument 3: Learning rate
    parser.add_argument('--learning_rate', type = float, default = 0.001, dest= 'lr',
                    help = 'Learning rate to use')  
    
     # Argument 4: Dropout probability
    parser.add_argument('--drop_p', type = float, default = 0.2,
                    help = 'Dropout probability') 

    # Argument 5: hidden units
    parser.add_argument('--hidden_units',type = int, default = 512,  nargs='+', dest = 'n',
                    help = 'Hidden units size ex: 512 256')  

    # Argument 6: Epochs
    parser.add_argument('--epochs', type = int, default = 5, 
                    help = 'Epochs number')  
    
    # Argument 7: CNN Architecture
    parser.add_argument('--arch', type = str, default = 'vgg19',  choices=['alexnet', 'vgg19', 'densenet121'],
                    help = 'Pretrainned CNN') 
    
    # Argument 8: Use GPU when available
    parser.add_argument('--gpu', type = str, default = 'y', choices=['y', 'n'],
                    help = 'Use GPU if available')

    return parser.parse_args()

