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

from utils import *

####### Parameter #######

LearningRate = 1e-4
BatchSize = 1000
print('Device: CUDA' if torch.cuda.is_available() else 'Device: cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################

parser = argparse.ArgumentParser(description='Some parameters for the model')

parser.add_argument('-ckp', '--ckp', type=str, default =None)
parser.add_argument('-art', '--artifact', type=str, default ='EMG')
parser.add_argument('-model', '--model_name', type=str, default ='DRNet')
parser.add_argument('-epoch', '--epochs', type=int, default = 100)

args = parser.parse_args()

########################

train_target = np.load(f'SC_pipeline/data/Train_{args.artifact}_target.npy')
train_noise = np.load(f'SC_pipeline/data/Train_{args.artifact}_noise.npy')
val_target = np.load(f'SC_pipeline/data/Val_{args.artifact}_target.npy')
val_noise = np.load(f'SC_pipeline/data/Val_{args.artifact}_noise.npy')

train_target = torch.from_numpy(train_target)
train_noise = torch.from_numpy(train_noise)
val_target = torch.from_numpy(val_target)
val_noise = torch.from_numpy(val_noise)

if args.model_name == 'DRNet':
    model = DRNet()

mkdir(f'SC_pipeline/{args.model_name}')
loss = nn.MSELoss()
optimizer = optim.Adam(params= model.parameters(), lr=LearningRate)

#######################

train_set = Data.TensorDataset(train_target, train_noise)
train_loader = Data.DataLoader(
    dataset = train_set,
    batch_size = BatchSize,
    shuffle = True
)

val_set = Data.TensorDataset(val_target, val_noise)
val_loader = Data.DataLoader(
    dataset = val_set,
    batch_size = BatchSize,
    shuffle = False
)

model.to(device)

######################

def model_train_art_clean_mix(model, model_name, train_loader, val_loader, optimizer, criterion, device, epochs, pretrained_ckp = None):
    
    min_val_loss = 1e10
    if pretrained_ckp != None:
        model.load_state_dict(torch.load(f'SC_pipeline/{model_name}/{pretrained_ckp}.pkl'))
        print(f'Load pretrained model: "{model_name}/{pretrained_ckp}.pkl"')
        min_val_loss = np.load(f'SC_pipeline/{model_name}/{pretrained_ckp}.npy')
        print(f'Pretrained model val_loss: {min_val_loss}')
        
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
        for step, (train_target, train_noise) in enumerate(train_loader):
        
            if (step//2)%2:
                train_input_1 = (train_target + train_noise)
            else:
                train_input_1 = train_target
            
            if ((step + 1)//2)%2:
                train_input_2 = (train_target + train_noise)
            else:
                train_input_2 = train_target
            
            train_input = torch.cat((train_input_1, train_input_2), dim = -1)
            train_target = torch.cat((train_target, train_target), dim = -1)
            
            std = torch.std(train_input)
            
            train_input = (train_input/std).float().to(device)
            train_target = (train_target/std).float().to(device)
            
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
            for step, (val_target, val_noise) in enumerate(val_loader):
            
                if (step//2)%2:
                    val_input_1 = (val_target + val_noise)
                else:
                    val_input_1 = val_target
                    
                if ((step + 1)//2)%2:
                    val_input_2 = (val_target + val_noise)
                else:
                    val_input_2 = val_target
                
                val_input = torch.cat((val_input_1, val_input_2), dim = -1)
                val_target = torch.cat((val_target, val_target), dim = -1)
                
                std = torch.std(val_input)
                
                val_input = (val_input/std).float().to(device)
                val_target = (val_target/std).float().to(device)
                
                val_output = model(val_input)
                val_loss = criterion(val_output, val_target)
                
                val_total_loss += val_loss.item()
            
        val_epoch_loss = val_total_loss / len(val_loader)
        val_loss_updating.append(val_epoch_loss)
        print('-----------------------')
        print('Epoch: ',epoch,' Train Loss: ',train_epoch_loss,' Val Loss: ',val_epoch_loss)
        
        if val_epoch_loss < min_val_loss :
            min_val_loss = val_epoch_loss
            print('Best validation performance ever, model saved.')
            torch.save(model.state_dict(), f'SC_pipeline/{model_name}/best_ckp.pkl')
            np.save(f'SC_pipeline/{model_name}/best_ckp.npy',min_val_loss)
            
    train_val_plot(train_loss_updating, val_loss_updating, save_loc=f'SC_pipeline/{model_name}/loss.jpg')
    plt.show()

model_train_art_clean_mix(model, args.model_name, train_loader, val_loader,
            optimizer, loss, device, args.epochs,
            args.ckp)