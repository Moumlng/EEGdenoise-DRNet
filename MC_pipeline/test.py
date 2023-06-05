import torch
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import scipy.io as io
import time

from utils import *
from network.DRNet import DRNet
import random

####### Parameter #######

print('Device: CUDA' if torch.cuda.is_available() else 'Device: cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dir = 'EC_Test/'

#######################

parser = argparse.ArgumentParser(description='Some parameters for the model')

parser.add_argument('-ckp', '--ckp', type=str, default = 'epoch1000')
parser.add_argument('-model', '--model_name', type=str, default='DRNet')
parser.add_argument('-index', '--index', type=int, default=None)

args = parser.parse_args()

if args.model_name == 'DRNet':
    model = DRNet()
    out_dir = test_dir + 'DRNet_output/'
    
mkdir(out_dir)

#######################



def model_test(model, model_name, device, test_dir, out_dir, test_num = 50, model_ckp = None):
    
    model.load_state_dict(torch.load(f'MC_pipeline/{model_name}/{model_ckp}.pkl'))
    model.eval().to(device)
    output_RRMSE = []
    output_CC = []
    trial_num = 0
    
    with torch.no_grad():
        for i in range(test_num):
            
            index = i + 1
            
            test_input_mat = io.loadmat(test_dir + f'sub{index}_raw.mat')
            test_input = test_input_mat['EEG_raw']
            test_input = torch.from_numpy(test_input)
            test_input = torch.permute(test_input, (2,0,1)).float().to(device)
            
            test_target_mat = io.loadmat(test_dir + f'sub{index}_clean.mat')
            test_target = test_target_mat['EEG_clean']
            test_target = torch.from_numpy(test_target)
            test_target = torch.permute(test_target, (2,0,1)).float().to(device)
            
            trial_num += test_target.shape[0]

            std = torch.std(test_input).float().to(device)
            test_input = torch.div(test_input, std)
            test_output = model(test_input)
            
            test_output = torch.mul(test_output, std)
            test_output = test_output.cpu().detach().numpy()
            test_target = test_target.cpu().detach().numpy()
            test_input = test_input.cpu().detach().numpy()
            
            
            for i in range(test_output.shape[0]):
                for j in range(test_output.shape[1]):
                    output_CC.append(np.corrcoef(test_output[i,j,:], test_target[i,j,:])[0,1])
                    output_RRMSE.append(RRMSE(test_output[i,j,:], test_target[i,j,:]))
                
            io.savemat(out_dir + f'sub{index}_out.mat', {'EEG_output':test_output, 'EEG_clean':test_target})
    
    output_RRMSE = np.array(output_RRMSE)
    output_CC = np.array(output_CC)
    output_RRMSE_mean = round(np.mean(output_RRMSE), 3)
    output_RRMSE_std = round(np.std(output_RRMSE), 3)
    output_CC_mean = round(np.mean(output_CC), 3)
    output_CC_std = round(np.std(output_CC), 3)
    
    print(f'Output RRMSE: {output_RRMSE_mean}±{output_RRMSE_std}')
    print(f'Output CC: {output_CC_mean}±{output_CC_std}')
    print(f'Test_nums:{trial_num}')

start_time = time.perf_counter()
model_test(model, args.model_name, device, test_dir, out_dir, 50, args.ckp)
end_time = time.perf_counter()
timecost = end_time - start_time
print(f'Time cost:{timecost}')

plt.show()