import torch
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import scipy.io
from scipy.fft import rfft, rfftfreq

from utils import *
from network.DRNet import DRNet
import random

####### Parameter #######

print('Device: CUDA' if torch.cuda.is_available() else 'Device: cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss = nn.MSELoss()
BatchSize = 1
threshold = 0.5

#######################

parser = argparse.ArgumentParser(description='Some parameters for the model')

parser.add_argument('-ckp', '--ckp', type=str, default = 'EMG100')
parser.add_argument('-art', '--artifact', type=str, default='EMG')
parser.add_argument('-model', '--model_name', type=str, default='DRNet')
parser.add_argument('-index', '--index', type=int, default=None)

args = parser.parse_args()

if args.model_name == 'DRNet':
    model = DRNet()

#######################

test_target = np.load(f'data/Test_{args.artifact}_target.npy')
test_noise = np.load(f'data/Test_{args.artifact}_noise.npy')

#Choose a number to plot
if args.index == None:
    test_index = chooseIndex(length = test_target.shape[0])
    
else:
    test_index = args.index

test_target_tensor = torch.from_numpy(test_target)
test_noise_tensor = torch.from_numpy(test_noise)


test_set = Data.TensorDataset(test_target_tensor, test_noise_tensor)
test_loader = Data.DataLoader(
    dataset = test_set,
    batch_size = BatchSize,
    shuffle = False
)

def model_test(model, model_name, test_loader, criterion, device, model_ckp = None):
    
    model.load_state_dict(torch.load(f'SC_pipeline/{model_name}/{model_ckp}.pkl'))
    model.eval().to(device)
    test_total_loss = []
    output_RRMSE = []
    output_CC = []
    outputs = []
    
    
    with torch.no_grad():
        for step, (test_target, test_noise) in enumerate(test_loader):
            
            if step % 2:
                test_input = test_target + test_noise
            else:
                test_input = test_target
            
            test_input = (test_input).float().to(device)
            test_target = (test_target).float().to(device)
            
            test_output = model(test_input)
            test_loss = criterion(test_output, test_target)
            
            outputs.append((test_output).squeeze(0).cpu().detach().numpy().astype('double'))
            
            test_total_loss.append(test_loss.item())
            output_RRMSE.append(RRMSE(test_input.cpu().detach().numpy(), test_target.cpu().detach().numpy()))
            output_CC.append(np.corrcoef(test_input.cpu().detach().numpy(), test_target.cpu().detach().numpy())[0,1])
            
    test_total_loss = np.array(test_total_loss)
    output_RRMSE = np.array(output_RRMSE)
    output_CC = np.array(output_CC)
    
    test_epoch_loss_mean = round(np.mean(test_total_loss), 3)
    test_epoch_loss_std = round(np.std(test_total_loss), 3)
    output_RRMSE_mean = round(np.mean(output_RRMSE), 3)
    output_RRMSE_std = round(np.std(output_RRMSE), 3)
    output_CC_mean = round(np.mean(output_CC), 3)
    output_CC_std = round(np.std(output_CC), 3)
    
    print(f'Model test loss: {test_epoch_loss_mean}±{test_epoch_loss_std}')
    print(f'Output RRMSE: {output_RRMSE_mean}±{output_RRMSE_std}')
    print(f'Output CC: {output_CC_mean}±{output_CC_std}')
    
    return outputs

Noise = test_noise[test_index]
Target = test_target[test_index]
Input = Target + Noise

model.load_state_dict(torch.load(f'SC_pipeline/{args.model_name}/{args.ckp}.pkl'))
Output1 = model(torch.from_numpy(Input).float().unsqueeze(0))

outputs = model_test(model, args.model_name, test_loader, loss, device, args.ckp)  
Output = Output1.squeeze(0).cpu().detach().numpy()
plot_numpy_set(Input, Target, Output, test_index)


'''
outputs = np.array(outputs)
np.savetxt(f'data/SC_output/csv/test_{args.artifact}_{args.model_name}_{args.ckp}_output.csv', outputs, delimiter = ',')
scipy.io.savemat(f'data/SC_output/mat/test_{args.artifact}_{args.model_name}_{args.ckp}.mat',{'input':(test_target + test_noise), 'target':test_target, 'output':outputs})
'''

#plt_spec_set(Input, Target, Output)

plt.figure(figsize=(12,4), dpi = 100)
fftfreq = rfftfreq(np.size(Input), 1/256)
input_fft = np.log10(abs(rfft(Input, np.size(Input))))
output_fft = np.log10(abs(rfft(Output, np.size(Output))))
target_fft = np.log10(abs(rfft(Target, np.size(Target))))
plt.plot(fftfreq, input_fft, label = 'raw', color = '#FF6600')
plt.plot(fftfreq, output_fft, label = 'output', color = '#003366')
plt.plot(fftfreq, target_fft, label = 'target', color = '#339966')
plt.ylabel('Amp(dB)')
plt.xlabel('freq/Hz')

plt.legend(loc = 'upper right')
plt.show()
