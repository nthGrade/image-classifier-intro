# Image Classifier Model

This project contains the code for training an image classifier model and predicting the name of a flower from an image.

## Files

1. `train.py`: This script is used to train a new network on a dataset. It prints out training loss, validation loss, and validation accuracy as the network trains.
2. `predict.py`: This script is used to predict the name of a flower from an image along with the probability of that name.
3. `model.py`: This file contains class and method definitions relating to this classifier model.
4. `utils.py`: This file contains utility functions for the model and scripts.
5. `cat_to_name.json`: This file contains a mapping of category labels to names. In this case, it is a map of flower names.
6. `config.yaml`: This file contains configuration details of the model to assist in training and prediction.

## Usage

### Training the Model

To train the model, run the following command:

```bash
python train.py /path/to/data --arch name_of_model --epochs num_epochs --save_dir /path/to/save/checkpoint
```

The script will start training the model using a given pre-trained model architecture, print out the training loss, validation loss, and validation accuracy as the network trains for the number of epochs, and then save a checkpoint of the model for testing or resuming training. 

This model currently supports the following pre-trained model architectures to start from:

* vgg19
* vgg13
* densenet161
* alexnet

The optional arguments of `--hidden_units`, `--learning_rate`, and `--epochs` can be used for tuning their corresponding hyperparameters, but can also be set in the `config.yaml`.

To execute the training on an available GPU, can add the optional argument `--gpu` when calling this script.

To resume training, can include the optional argument `--resume` followed by tthe path to the .pth checkpoint file to be loaded.

### Predicting Flowers

To predict the flower name from an image, run the following command:

```bash
python predict.py /path/to/image /path/to/checkpoint
```

The script will output the name of the flower and the probability of that name.

The optional argument `--top_k` followed by an integer can adjust the number of predicted values the classifier should display. Otherwise, this parameter is set based on its value in `config.yaml`.

The optional argument `--category_names` allows the user to point the model to a JSON mapping file for label matching other than `cat_to_name.json`. 

To compute the prediction(s) on an available GPU, can add the optional argument `--gpu` when calling this script.
