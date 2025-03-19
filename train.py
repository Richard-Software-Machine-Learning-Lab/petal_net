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
import json
import argparse

# python train.py --data_dir "./flowers/"
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="./flowers/", help='')
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='')
parser.add_argument('--arch', type=str, default='vgg11', help='')
parser.add_argument('--learning_rate', type=float, default=.001, help='')
parser.add_argument('--hidden_layer', type=int, default=25088, help='')
parser.add_argument('--gpu',  type=str, default='gpu', help='')
parser.add_argument('--epochs', type=int, default=10, help='')
parser.add_argument('--dropout', type=float, default=0.2, help='')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f) 


def load_data(data_dir_):
    data_dir = data_dir_
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    dataloaders = {'train':trainloader, 'valid':validloader, 'test': testloader}
    dataset_sizes = {'train': len(train_data), 'valid': len(valid_data), 'test': len(test_data)}
    return  train_data, dataloaders, dataset_sizes
    

def device_(gpu_):
    if gpu_ == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device
    
def classifier(arch_,gpu_, hidden_layer_, dropout_, learning_rate_):

    device = device_(gpu_)
    hidden_layer = hidden_layer_
    dropout = dropout_
    learning_rate = learning_rate_
    
    
    if arch_ == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch_ == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    num_out = len(cat_to_name)

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(hidden_layer, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(4096, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(256, num_out))

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, gpu):
    best_model_ = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('*******************************')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            loss = 0.0
            correct_predictions = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device_(gpu))
                labels = labels.to(device_(gpu))

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss += loss.item() * inputs.size(0)
                correct_predictions += torch.sum(preds == labels.data)

            epoch_loss = loss / dataset_sizes[phase]
            epoch_accuracy = correct_predictions.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss} and Accuracy: {epoch_accuracy}')

            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_ = copy.deepcopy(model.state_dict())
                
    print(f'Best Accuracy: {best_accuracy}')
    model.load_state_dict(best_model_)
    return model    

def save_checkpoint(model, train_data, optimizer, epochs, save_dir, arch, learning_rate, hidden_layer, gpu, dropout):

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'class_to_idx':model.class_to_idx,
              'classifier':model.classifier,
              'state_dict':model.state_dict(),
              'optim':optimizer.state_dict(),
              'epochs':epochs,
              'arch': arch,
              'learning_rate': learning_rate,
              'hidden_layer': hidden_layer,
              'gpu': gpu,
              'dropout': dropout}

    torch.save(checkpoint, save_dir)
    
if __name__ == "__main__":
    console_inputs = parser.parse_args()
    data_dir = console_inputs.data_dir
    save_dir = console_inputs.save_dir
    arch = console_inputs.arch
    learning_rate = console_inputs.learning_rate
    hidden_layer = console_inputs.hidden_layer
    gpu = console_inputs.gpu
    epochs = console_inputs.epochs
    dropout = console_inputs.dropout
    train_data, dataloaders, dataset_sizes = load_data(data_dir)
    
    model, criterion, optimizer = classifier(arch, gpu, hidden_layer, dropout, learning_rate)
    model = train_model(model, criterion, optimizer, epochs, dataloaders, dataset_sizes, gpu)
            
    
    save_checkpoint(model, train_data, optimizer, epochs, save_dir, arch, learning_rate, hidden_layer, gpu, dropout)
            
    

    

