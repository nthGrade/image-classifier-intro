#!/usr/bin/env python3
###
# PROGRAMMER: Nathan G.
# DATE CREATED: 10/09/2023
# DATE REVISED: 10/21/2023
# PURPOSE: Predict flower name from an image along with the
#          probability of that name. 
#          Prints the flower name and class probability.
# 
# Basic Usage: python predict.py /path/to/image checkpoint
# Options: 
# * Return top K most likely classes: 
#   $ python predict.py input checkpoint --top_k 3 
# * Use a mapping of categories to real names: 
#   $ python predict.py input checkpoint --category_names cat_to_name.json 
# * Use GPU for inference: 
#   $ python predict.py input checkpoint --gpu
import argparse
import torch
from model import ClassifierModel
from PIL import Image
from utils import category_map, load_config, process_image

def predictions(model, image, top_k, device):
        ''' Predict the class (or classes) of an image using a trained deep learning model.

            Args:
                model: Classifier model to use.
                image: Tensor for a PIL image.
                top_k: How many of the top probabilities to produce.
                device: Torch device to run the model on.

            Returns:
                topk_p: probabilities returned by the model.
                topk_classes: class names returned by the model.
            
        '''

        # If GPU is available, run on that, otherwise run on CPU.
        model.to(device)

        # Ensure the model is in eval mode
        model.eval()

        # Make predictions on the image
        output = model(image)

        # Grab class_to_idx info and invert the dict for index to class mapping
        class_to_idx = model.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Convert logarithmic output to probabilities
        ps = torch.exp(output)

        # Find the top_k predictions
        topk_p, topk_indices = torch.topk(ps, top_k)

        # Convert tensors to lists
        topk_p = topk_p.tolist()[0]
        topk_indices = topk_indices.tolist()[0]

        # Get the class labels
        topk_classes = [idx_to_class[idx] for idx in topk_indices]

        return topk_p, topk_classes

def main():
    parser = argparse.ArgumentParser(description='For using Image Classifier')

    parser.add_argument('input', type=str,
                        help='Path to an image to classify.')
    parser.add_argument('checkpoint', type=str,
                        help='Path to a model checkpoint to load.')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to the configuration file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Top-K predicted values the classifier should display.')
    parser.add_argument('--category_names', default='cat_to_name.json', type=str,
                        help='Path to JSON file with dictionary of category labels to category names.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Determine whether to load tensors to GPU or not.')
    args = parser.parse_args()

    # For switch to cuda if requested and able to
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Preprocess the image
    img = Image.open(args.input)
    pil_image = process_image(img)
    img_t = torch.from_numpy(pil_image).type(torch.FloatTensor).unsqueeze(dim=0)
    img_t.to(device)

    # Extract category to labels mapping
    cat_to_name = category_map(args.category_names)

    # Load config yaml
    loaded_config = load_config(args.config)

    # Initialize the model
    arch = loaded_config['model']['arch']
    input_size = loaded_config['model']['input_size']
    output_size = loaded_config['model']['output_size']
    hidden_units = loaded_config['model']['hidden_layers']
    dropout = loaded_config['model']['dropout']
    learn_rate = loaded_config['training']['learning_rate']

    new_model = ClassifierModel(arch, input_size, output_size, hidden_units, dropout, learn_rate)
    optimizer = torch.optim.Adam(new_model.model.classifier.parameters(), lr=learn_rate)

    # Load the checkpoint into the model
    loaded_model = new_model.load_checkpoint(args.checkpoint, optimizer)
    
    # Make predictions
    print("Thinking... ")
    probs, classes = predictions(loaded_model, img_t, args.top_k, device)

    # Convert classes to names
    flower_names = [cat_to_name[str(c)] for c in classes]

    # Print the predicted result(s) and corresponding class probability
    print(f"These are the top {args.top_k} predicted results: ")
    print(f'{"Flower Names":<10}    {"Probabilities":<10}')
    for flower, prob in zip(flower_names, probs):
        print(f"{flower:<10}    {100 * prob:<10}")

if __name__ == "__main__":
    main()