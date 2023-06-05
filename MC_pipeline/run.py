import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.fft import rfft, rfftfreq
import argparse
import time

from utils import *
from network.DRNet import DRNet
import random

####### Parameter #######

print('Device: CUDA' if torch.cuda.is_available() else 'Device: cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Some parameters for the model')

parser.add_argument('-ckp', '--ckp', type=str, default = 'epoch1000')
parser.add_argument('-model', '--model_name', type=str, default='DRNet')
parser.add_argument('-in', '--input_name', type=str, default='10_raw_60')
parser.add_argument('-out', '--output_name', type=str, default = None)

args = parser.parse_args()

if args.model_name == 'DRNet':
    model = DRNet()

#######################

def inference(input, device, model, model_name, model_ckp):
    
    #assert input.shape[0] == 60
    #input = standardize(input)
    #print(f'input shape : {str(input.shape)}')
    with torch.no_grad():
        
        input = torch.from_numpy(input).unsqueeze(0).float().to(device)
        
        model.to(device)
        model.eval()
        model.load_state_dict(torch.load(f'MC_pipeline/{model_name}/{model_ckp}.pkl'))
        input = model(input)
        
        output = input.squeeze(0).cpu().detach().numpy()
        print('Input length:',output.shape[1] / 250)
        
    return output
    

startTime = time.perf_counter()
Input_raw = np.loadtxt(f'MC_pipeline/data/input/{args.input_name}.csv', delimiter = ',')
ChannelIndex = chooseIndex(Input_raw.shape[0])
ex_input = Input_raw[ChannelIndex][0:512]
Output = inference(Input_raw, device, model, args.model_name, args.ckp)
endTime = time.perf_counter()

TimeCost = (endTime-startTime)
print(f'Time Cost: {TimeCost}s.')

if args.output_name != None:
    np.savetxt(f'MC_pipeline/data/output/csv/run_{args.output_name}.csv', Output, delimiter=',')
    scipy.io.savemat(f'MC_pipeline/data/output/mat/run_{args.output_name}.mat',{'input':Input_raw, 'output': Output})