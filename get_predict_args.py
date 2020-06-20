#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  29 May  2020                                 
# REVISED DATE :  29 May  2020
# PURPOSE: Create a function that retrieves the args for the predict script
#          from the user using the Argparse Python module. If the user does not 
#          input the params default value is used. (Based on clasroom first project)
# 
# Command Line Arguments:
# 
#  1. Data image path                           --data_dir with default value 'flowers'
#  2. checkpoint path                           --checkpoint
#  3. Top K probabilities                       --top_k
#  4. Categroy Names                            --category_names
#  5. GPU to specified gpu resources use        --gpu with default value 'y'
##

# Imports python modules

import os
import argparse


# 
def get_predict_args():
    '''
    Retrieves and parses the 5 command line arguments provided by the user from 
    command line. argparse module is used. 
    If some arguments is missing default value is used. 
    Command Line Arguments:
    
    Args:
        1. Data image path                           --data_dir with default value 'flowers'
        2. checkpoint path                           --checkpoint
        3. Top K probabilities                       --top_k
        4. Categroy Names                            --category_names
        5. GPU to specified gpu resources use        --gpu with default value 'y'

    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    '''
    
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser('python predict.py',description='Predict flower name', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
       
    # Argument 1: path to flower image 
    parser.add_argument('data_dir', type = str, default= 'flower',
                    help = 'path to the image')  
    
    # Argument 2: checkpoint 
    parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth', 
                    help = 'path to the checkpoint')    
    
    # Argument 3: Top k
    parser.add_argument('--top_k', type = int, default = 3,
                    help = 'Top k classes')  

    # Argument 4: Category Names
    parser.add_argument('--category_names',type = str, default = 'cat_to_name.json',
                    help = 'Category names')  

    # Argument 5: Use GPU when available
    parser.add_argument('--gpu', type = str, default = 'y', choices=['y', 'n'],
                    help = 'Use GPU if available')

    return parser.parse_args()


