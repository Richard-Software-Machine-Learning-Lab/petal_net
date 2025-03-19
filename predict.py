import torchvision
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch import optim
import torch.utils.data as data
import torchvision.models as models
import os
import copy
import time
import matplotlib.pyplot as plt
from utils import active_session
import json
import argparse
import torch.nn.functional as F
from PIL import Image

# python predict.py --image_file "flowers/test/1/image_06743.jpg"
parser = argparse.ArgumentParser()

parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='')
parser.add_argument('--image_file', type=str, default='flowers/test/1/image_06743.jpg', help='')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth', help='')
parser.add_argument('--topk', type=int, default=5, help='')
parser.add_argument('--gpu', default='gpu', type=str, help='')


def device_(gpu_):
    if gpu_ == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
    

def process_image(image):
    image_processing= transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image_loaded = Image.open(image)
    image_tensor = (image_processing(image_loaded))
    return image_tensor

    
def load_checkpoint(checkpoint_path): 
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    lr = checkpoint['learning_rate']
    hidden_layer = checkpoint['hidden_layer']
    gpu = checkpoint['gpu']
    epochs = checkpoint['epochs']
    dropout = checkpoint['dropout']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
    
    if arch == 'vgg11':
       model = models.vgg11(pretrained=True)
    elif arch == 'vgg16':
       model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.require_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']  
    return model

def predict(device, image_path, model, topk):
    
    model.to(device)
    image_torch = process_image(image_path)
    image_torch = image_torch.unsqueeze_(0)
    image_torch = image_torch.float()
    
    with torch.no_grad():
        output = model(image_torch.to(device))
        
    prediction = F.softmax(output.data,dim=1)  
    probabilities, index = prediction.topk(topk)
    probabilities_ = probabilities.cpu().numpy()[0]
    indices = index.cpu().numpy()[0]
    
    index_class = {value:key for key, value in model.class_to_idx.items()}
    classes = [index_class[index] for index in indices]
    return probabilities_, classes



def results(cat_to_name, device, image_file, model, topk):
    print("****** Waiting results ********")
    probabilities_, classes = predict(device, image_file, model, topk)
    flower_names = [cat_to_name[str(each_class)] for each_class in classes]
    maxpos = probabilities_.argmax()
    print("Most likely image class and it's associated probability")
    print(f'Class: {flower_names[maxpos]} and Probability: {probabilities_[maxpos]}')
    print(f"Top {topk} classes and probabilities")
    top_classes_probabilities = dict(zip(flower_names, probabilities_))
    print(top_classes_probabilities)
    
if __name__ == "__main__":
    
    console_inputs = parser.parse_args()
    json_file = console_inputs.json_file
    image_file = console_inputs.image_file
    checkpoint_path = console_inputs.checkpoint_file
    topk = console_inputs.topk
    gpu = console_inputs.gpu
    device = device_(gpu)
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f) 

    model = load_checkpoint(checkpoint_path)
    results(cat_to_name, device, image_file, model, topk)
    

   
    
    
    
            