#!/usr/bin/env python3
###
# PROGRAMMER: Nathan G.
# DATE CREATED: 10/09/2023
# DATE REVISED: 10/21/2023
# PURPOSE: Trains a new network on a data set.
#          Prints out training loss, validation loss, 
#          and validation accuracy as the network trains
# 
# Basic Usage: python train.py data_directory
# Options:
# * Set directory to save checkpoints:
#    $ python train.py data_dir --save_dir save_directory 
# * Choose architecture: 
#    $ python train.py data_dir --arch "vgg13" 
# * Set hyperparameters: 
#    $ python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
# * Use GPU for training: 
#    $ python train.py data_dir --gpu
import argparse
import torch
from model import ClassifierModel
from utils import load_config, load_data

def training(model, epochs, optimizer, trainloader, validloader, device):
        ''' Train model for number of epochs on the given DataLoader sets.

            Args:
                model: Classifier model to be trained.
                epochs: Number of passes through the entire dataset.
                optimizer: Selected algorithm to tune parameters for minimizing loss function
                trainloader: DataLoader object for the training image dataset.
                validloader: DataLoader object for the validation image dataset.
                device: Torch device to run the model on.

            Returns:
                N/A

        '''
        # Set additional hyperparameters for training
        criterion = torch.nn.NLLLoss()

        # If GPU is available, run on that, otherwise run on CPU.
        model.to(device)

        steps = 0
        running_loss = 0
        print_every = 5
        model.train()

        for epoch in range(epochs):
            for images_tn, labels_tn in trainloader:
                steps += 1
                # make sure to move these tensors to the correct device
                images_tn, labels_tn = images_tn.to(device), labels_tn.to(device)

                # training loop starts here
                optimizer.zero_grad()

                # get output probabilities
                logps_tn = model.forward(images_tn)

                # get loss
                loss_tn = criterion(logps_tn, labels_tn)

                # backprop the loss through the system
                loss_tn.backward()
                optimizer.step()
                running_loss += loss_tn.item()

                # do validation loop
                if steps % print_every == 0:
                    model.eval()

                    test_loss = 0
                    accuracy = 0

                    with torch.no_grad():
                        for images_vld, labels_vld in validloader:
                            # move these tensors to correct device
                            images_vld, labels_vld = images_vld.to(device), labels_vld.to(device)

                            logps_vld = model.forward(images_vld)
                            batch_loss = criterion(logps_vld, labels_vld)
                            test_loss += batch_loss.item()

                            # calculate accuracy
                            ps = torch.exp(logps_vld)
                            top_ps, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels_vld.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Training loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {test_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    
                    running_loss = 0
                    model.train()
        
        print('Training completed!')

def test_classifier(model, testloader, device):
        ''' Using a DataLoader for testing, evaluate the accuracy of the current model.

            Args:
                testloader: DataLoader object for the test image dataset.
                device: Torch device to run the model on.

            Returns:
                N/A
        '''
        # Put model into evaluation mode
        model.eval()
        correct = 0
        total = 0

        # Do test validation loop
        with torch.no_grad():
            for images_tst, labels_tst in testloader:
                # move these tensors to the correct device
                images_tst, labels_tst = images_tst.to(device), labels_tst.to(device)

                # run test images through network to get predictions
                outputs = model(images_tst)
                _, predicted = torch.max(outputs, 1)

                # add to test results
                total += labels_tst.size(0)
                correct += (predicted == labels_tst).sum().item()
        
        accuracy = 100 * (correct / total)
        print(f"Accuracy of the network on the test images: {accuracy:.3f}%")

        # Put model back into training mode
        model.train()

def main():
    # Create a separate parser for the config file
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to the configuration file.')
    args, _ = config_parser.parse_known_args()

    # Load model config
    loaded_config = load_config(args.config)

    # Now create the main parser
    parser = argparse.ArgumentParser(description='For training Image Classifier')
    parser.add_argument('data_dir', type=str,
                        help='Path to folder containing training, validation, and test data.')
    parser.add_argument('--arch', default=None, type=str,
                        help='Choose model architecture to use.')
    parser.add_argument('--hidden_units', default=None, type=int,
                        help='Number of hidden layer units to use in classifier.')
    parser.add_argument('--learning_rate', default=loaded_config['training']['learning_rate'], type=float,
                        help='Learning rate to use for optimizer when training.')
    parser.add_argument('--epochs', default=loaded_config['training']['epochs'], type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--save_dir', default='checkpoints', type=str,
                        help='Set directory to save checkpoints.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Determine whether to load tensors to GPU or not.')
    parser.add_argument('--resume', type=str,
                        help='Use an already saved checkpoint to load and continue training.')
    args = parser.parse_args()

    # Switch to cuda if able to
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Directory setup
    data_dir = args.data_dir
    batch_size = loaded_config['data']['batch_size']
    save_dir = args.save_dir

    # Load the datasets for training
    data = load_data(data_dir, batch_size)

    # Model setup
    # Find the correct model config
    model_config = None
    for model in loaded_config['models']:
        if model['arch'] == args.arch:
            model_config = model
            break

    if model_config is None:
        print(f"No configuration found for architecture {args.arch}.")
        return     
    
    arch = args.arch
    input_size = model_config['input_size']
    output_size = model_config['output_size']
    hidden_units = args.hidden_units
    dropout = model_config['dropout']
    learn_rate = args.learning_rate
    epochs = args.epochs
    if args.hidden_units is None:
        hidden_units = model_config['hidden_layers']
    else:
        hidden_units = args.hidden_units

    new_model = ClassifierModel(arch, input_size, output_size, hidden_units, dropout, learn_rate)
    optimizer = torch.optim.Adam(new_model.model.classifier.parameters(), lr=learn_rate)

    if args.resume is not None:
        # Load the checkpoint if available to continue training/testing
        print("Loading checkpoint...")
        new_model.model = new_model.load_checkpoint(args.resume, optimizer)
    
    # Complete training loop for number of epochs
    print("Training Start!")
    training(new_model.model, epochs, optimizer, data['train']['loader'], data['valid']['loader'], device)

    # Test the model accuracy
    print("Evaluating against test dataset... ")
    test_classifier(new_model.model, data['test']['loader'], device)

    # Save the current model as a checkpoint
    print("Saving checkpoint now... ")
    new_model.save_checkpoint(save_dir, data['train']['dataset'], optimizer)

if __name__ == "__main__":
    main()