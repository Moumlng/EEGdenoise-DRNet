import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.fft import rfft, rfftfreq
import argparse
import time

from utils import *
from network.DRNet import DRNet

####### Parameter #######

print('Device: CUDA' if torch.cuda.is_available() else 'Device: cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Some parameters for the model')

parser.add_argument('-ckp', '--ckp', type=str, default = 'EMG100')
parser.add_argument('-model', '--model_name', type=str, default='DRNet')
parser.add_argument('-in', '--input_name', type=str, default='59channel_1_raw')
parser.add_argument('-out', '--output_name', type=str, default = None)

args = parser.parse_args()

if args.model_name == 'DRNet':
    model = DRNet()

#######################

def inference(input, device, model, model_name, model_ckp):
    
    assert len(input.shape) == 1
    input = torch.from_numpy(input).unsqueeze(0).float().to(device)
    
    model.to(device).eval()
    model.load_state_dict(torch.load(f'SC_pipeline/{model_name}/{model_ckp}.pkl'))
    output = model(output)
    
    output = output.squeeze(0).cpu().detach().numpy()
    
    return output
    

startTime = time.perf_counter()
Input_raw = np.loadtxt(f'SC_pipeline/data/input/{args.input_name}.csv', delimiter = ',')

if (len(Input_raw.shape)) == 1:
    Input_raw = np.expand_dims(Input_raw, dim = 0)
    
ChannelIndex = chooseIndex(Input_raw.shape[0])
ex_input = Input_raw[ChannelIndex][0:512]

Output = np.empty_like(Input_raw)

for step,channel in enumerate(Input_raw):
    
    temp_output = inference(channel, device, model, args.model_name, args.ckp)
    Output[step] = temp_output

endTime = time.perf_counter()

TimeCost = (endTime-startTime)
print(f'Time Cost: {TimeCost}s.')

if args.output_name != None:
    
    mkdir('SC_pipeline/data/output/csv')
    mkdir('SC_pipeline/data/output/mat')
    np.savetxt(f'SC_pipeline/data/output/csv/run_{args.output_name}.csv', Output, delimiter=',')
    scipy.io.savemat(f'SC_pipeline/data/output/mat/run_{args.output_name}.mat',{'input':Input_raw, 'output': Output})