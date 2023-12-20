#!/usr/bin/env python3
###
# PROGRAMMER: Nathan G.
# DATE CREATED: 10/09/2023
# DATE REVISED: 10/21/2023
# PURPOSE: Class and method definitions relating to the classifier model
# 
import torch
from torch import nn, optim
import torchvision.models as models
from collections import OrderedDict
import datetime

# Define some models
model_funcs = {
    'alexnet': models.alexnet,
    'densenet161': models.densenet161,
    'vgg13': models.vgg13,
    'vgg19': models.vgg19
}

models_dict = {name: func(weights="DEFAULT") for name, func in model_funcs.items()}

class ClassifierModel(nn.Module):
    def __init__(self, model_name, input_size, output_size, hidden_units, dropout, learn_rate):
        # Initialize the model
        super(ClassifierModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.model_name = model_name
        self.model = self.create_model(model_name, input_size, hidden_units, output_size, dropout)

    def create_model(self, model_name, input_size, hidden_units, output_size, dropout):
        ''' Create the classifier model from a pretrained network for feature finding and 
            a custom feed-forward network for the classifier. 

            Args:
                model_name: String of the model architecture name.
                input_size: In_features for the model classifier.
                hidden_units: Number of hidden layer units.
                output_size: Out_features for the model classifier.
                dropout: Percentage of neurons to ignore during training per layer.

            Returns:
                pt_model: configured model with new classifier.
        '''

        # Load a pretrained network for obtaining features.
        pt_model = models_dict[model_name]
        
        # Freeze parameters so we don't backprop through them.
        for param in pt_model.parameters():
            param.requires_grad = False
        
        # Define a new, untrained feed-forward network as the classifier.
        hidden_mid_units = round(hidden_units, -3) // 4    # should be 1000 when using default hidden_units

        pt_model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_units)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(hidden_units, hidden_mid_units)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(p=dropout)),
            ('fc3', nn.Linear(hidden_mid_units, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        return pt_model

    def forward(self, input):
        ''' Runs an input image tensor through the model.

            Args:
                input: tensor for a PIL image.

            Returns:
                output: probability results from the network.
        '''
        # pass the input into the model's forward method to get output
        output = self.model.forward(input)
        return output
    
    def save_checkpoint(self, save_dir, train_data, optimizer):
        ''' Saves a checkpoint of the model's training and the optimizer used.

            Args:
                save_dir: directory to save checkpoints in.
                train_data: training data used.
                optimizer: optimizer algorithm used during training.

            Returns:
                N/A
        '''
        # Add attribute for class to indices mapping
        self.class_to_idx = train_data.class_to_idx

        # Save the model as a checkpoint
        checkpoint = {
            'model_arch': self.model_name,
            'classifier': self.model.classifier,
            'optimizer_state_dict': optimizer.state_dict(),
            'learn_rate': self.learn_rate,
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx
        }
        current_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        torch.save(checkpoint, save_dir + '/' + current_date + '_' + self.model_name +
                    '_image_classifier_checkpoint-ng.pth')
    
    def load_checkpoint(self, filepath, optimizer):
        ''' Loads a model checkpoint for use by this classifieir.

            Args:
                filepath: String of /path/to/checkpoint.pth
                optimizer: optimizer algorithm used for training.
            
            Returns:
                loaded_model: instance of ClassifierModel with checkpoint attributes loaded.
        '''
        checkpoint = torch.load(filepath)
        
        # initialize the same model architecture that was used before
        # For properly setup checkpoints moving forward
        loaded_model = models_dict[checkpoint['model_arch']]
        
        # Freeze parameters (again)
        for param in loaded_model.parameters():
            param.requires_grad = False
        
        # Load the state dict for the previous model
        loaded_model.classifier = checkpoint['classifier']
        loaded_model.load_state_dict(checkpoint['state_dict'])
        loaded_model.class_to_idx = checkpoint['class_to_idx']
        
        # Load an optimizer
        # Accommodate prev checkpoint version for testing
        optimizer = optim.Adam(loaded_model.classifier.parameters(), lr=checkpoint['learn_rate'])
        
        # Update optimizer with state_dict for previous model's optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return loaded_model  
