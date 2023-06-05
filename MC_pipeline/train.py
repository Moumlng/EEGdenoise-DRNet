import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import argparse

from network.DRNet import DRNet
from dataset import EEGdataset

from utils import *

####### Parameter #######

LearningRate = 1e-4
BatchSize = 500
print('Device: CUDA' if torch.cuda.is_available() else 'Device: cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################

parser = argparse.ArgumentParser(description='Some parameters for the model')

parser.add_argument('-ckp', '--ckp', type=str, default =None)
parser.add_argument('-model', '--model_name', type=str, default ='DRNet')
parser.add_argument('-epoch', '--epochs', type=int, default = 1000)

args = parser.parse_args()

########################

if args.model_name == 'DRNet':
    model = DRNet()

mkdir(f'MC_pipeline/{args.model_name}')
loss = nn.MSELoss()
optimizer = optim.Adam(params= model.parameters(), lr=LearningRate)

#######################

train_set = EEGdataset(nums_file = 262)
train_loader = Data.DataLoader(
    dataset = train_set,
    batch_size = BatchSize,
    shuffle = True
)

val_set = EEGdataset(nums_file = 30, bias = 262)
val_loader = Data.DataLoader(
    dataset = val_set,
    batch_size = BatchSize,
    shuffle = False
)

model.to(device)

######################

def model_train(model, model_name, train_loader, val_loader, optimizer, criterion, device, epochs, pretrained_ckp = None):
    
    min_val_loss = 1e10
    if pretrained_ckp != None:
        model.load_state_dict(torch.load(f'MC_pipeline/{model_name}/{pretrained_ckp}.pkl'))
        print(f'Load pretrained model: "MC_pipeline/{model_name}/{pretrained_ckp}.pkl"')
        min_val_loss = np.load(f'MC_pipeline/{model_name}/{pretrained_ckp}.npy')
        print(f'Pretrained model val_loss: MC_pipeline/{min_val_loss}')
    
    train_loss_updating = []
    val_loss_updating = []
    
    for epoch in range(epochs):
        
        train_loss = 0.0
        train_total_loss = 0.0
        train_epoch_loss = 0.0
        val_loss = 0.0
        val_total_loss = 0.0
        val_epoch_loss = 0.0
        
        model.train().to(device)
        for step, (train_clean, train_raw) in enumerate(train_loader):
            
            train_input = train_raw.float().to(device)
            train_target = train_clean.float().to(device)
            
            std = torch.std(train_input).float().to(device)
            
            train_input = torch.div(train_input, std)
            train_target = torch.div(train_target, std)
            
            optimizer.zero_grad()
            train_output = model(train_input)
            train_loss = criterion(train_output, train_target)
            train_loss.backward()
            optimizer.step()
            
            train_total_loss += train_loss.item()
        
        train_epoch_loss = train_total_loss / len(train_loader)
        train_loss_updating.append(train_epoch_loss)
        
        model.eval().to(device)
        with torch.no_grad():
            for step, (val_clean, val_raw) in enumerate(val_loader):
                
                val_input = val_raw.float().to(device)
                val_target = val_clean.float().to(device)
                
                std = torch.std(val_input).float().to(device)
                
                val_input = torch.div(val_input, std)
                val_target = torch.div(val_target, std)
                
                val_output = model(val_input)
                val_loss = criterion(val_output, val_target)
                
                val_total_loss += val_loss.item()
            
        val_epoch_loss = val_total_loss / len(val_loader)
        val_loss_updating.append(val_epoch_loss)
        print('-----------------------')
        print('Epoch: ',epoch,' Train Loss: ',train_epoch_loss,' Val Loss: ',val_epoch_loss)
        
        if val_epoch_loss < min_val_loss :
            min_val_loss = val_epoch_loss
            print('Best performance ever, model saved.')
            torch.save(model.state_dict(), f'MC_pipeline/{model_name}/best_ckp.pkl')
            np.save(f'MC_pipeline/{model_name}/best_ckp.npy',min_val_loss)
        
    train_val_plot(train_loss_updating, val_loss_updating, save_loc=f'MC_pipeline/{model_name}/loss.jpg')
    
    #plot_valLoader_set(val_input, val_target, val_output, index = 5)
    #print(np.size(val_output.cpu().detach().numpy()))
    #print(np.size(val_output.cpu().detach().numpy()[0]))
    #print(len(val_loader))
    
    plt.show()

model_train(model, args.model_name, train_loader, val_loader,
            optimizer, loss, device, args.epochs,
            args.ckp)