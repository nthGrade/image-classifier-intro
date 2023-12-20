#!/usr/bin/env python3
###
# PROGRAMMER: Nathan G.
# DATE CREATED: 10/09/2023
# DATE REVISED: 10/21/2023
# PURPOSE: Utility functions for the model
#
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import json
import os
import yaml

def category_map(json_file):
    ''' Load in a file mapping from category label to category name.

        Args:
            json_file: file containing dictionary mapping of labels and names.

        Returns:
            cat_to_name: dictionary mapping labels to names
    '''

    with open(json_file, 'r') as f:
        try:
            cat_to_name = json.load(f)
        except json.JSONDecodeError as err:
            print(err)
    
    return cat_to_name

def load_config(yaml_file):
    ''' Load in a yaml file for initial model settings.

        Args:
            yaml_file: a config file of model, training, predicting, and data info

        Returns:
            config: dictionary of info gathered from yaml_file
    '''
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(err)
    
    return config

def load_data(data_dir, batch_size=64):
    ''' Loads and transforms provided dataset for use in training and testing.

        Args:
            data_dir: Root directory for image dataset. Assumes dir has /train, /valid, 
            and /test subfolders.
            batch_size: Size of batches to use in dataloaders. Defaults to 64.
        
        Returns:
            A dictionary containing ImageFolder and DataLoader objects for training, validation,
            and testing.
    '''
    # Initialize directory paths
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Make sure the directories exist
    for dir in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(dir):
             raise FileNotFoundError(f"Directory {dir} does not exist.")

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
      transforms.RandomRotation(30),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # validation and testing sets use the same transforms
    test_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the tranforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Create a dictionary to store the data and dataloaders
    data_dict = {
        'train': {'dataset': train_data, 'loader': trainloader},
        'valid': {'dataset': valid_data, 'loader': validloader},
        'test': {'dataset': test_data, 'loader': testloader}
    }

    return data_dict

def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.

        Args:
            pil_image: A PIL image.
        
        Returns:
            A NumPy array of the preprocessed image.
    '''
    
    if not isinstance(pil_image, Image.Image):
        raise ValueError('Input image must be a PIL image.')
    
    # Ensure image is RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Resize and crop the image
    aspect_ratio = pil_image.size[0] / pil_image.size[1]
    if aspect_ratio > 1:
        pil_image = pil_image.resize((int(256 * aspect_ratio), 256))
    else:
        pil_image = pil_image.resize((256, int(256 / aspect_ratio)))

    width, height = pil_image.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert to NumPy array for easier color channel encoding
    np_image = np.array(pil_image) / 255

    # Normalize the image with expected mean and standard deviation values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder the image array in preparation for classification: color channel should be first
    # instead of third.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
