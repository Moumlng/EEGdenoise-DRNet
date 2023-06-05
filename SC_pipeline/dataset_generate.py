import numpy as np
import matplotlib.pyplot as plt
import random
import time
from utils import *

## Parameter ##
Train_num = 40000
Val_num = 5000
Test_num = 5000
SNR_min = -2
SNR_max = 2
artifact = 'EMG' # 'EOG' or 'EMG'
################

EEG=np.load('SC_pipeline/data/EEG_all_epochs.npy')
EMG=np.load('SC_pipeline/data/EMG_all_epochs.npy')
EOG=np.load('SC_pipeline/data/EOG_all_epochs.npy')

print('EEG Shape:',EEG.shape)
print('EMG Shape:',EMG.shape)
print('EOG Shape:',EOG.shape)

if artifact == 'EMG':
    ART = EMG
    
elif artifact == 'EOG':
    ART = EOG
    
#########   RAW EXAMPLE   ##########

plt.figure(0)

x = np.linspace(0,2,512)
print(x)

plt.subplot(3,1,1)
plt.plot(x,EEG[0])
plt.xlabel('time/s')
plt.ylabel('amp')
plt.legend(['EEG'], loc = 'upper right')

plt.subplot(3,1,2)
plt.plot(x,EMG[0])
plt.xlabel('time/s')
plt.ylabel('amp')
plt.legend(['EMG'], loc = 'upper right')

plt.subplot(3,1,3)
plt.plot(x,EOG[0])
plt.xlabel('time/s')
plt.ylabel('amp')
plt.legend(['EOG'], loc = 'upper right')

plt.show()

################################

def IOset(EEG, ART, num, SNRmin, SNRmax):
    
    random.seed(time.time())
    
    #SNR randomed
    SNR_Train_dB = np.random.uniform(SNRmin, SNRmax, (num))
    SNR_Train = 10**(0.1*(SNR_Train_dB)).reshape(SNR_Train_dB.shape[0])
    #print('SNR Size:', SNR_Train.shape)
    
    NoiseSet = []
    TargetSet = []
    
    for i in range(num):
        
        CleanEEG = EEG[random.randint(0,EEG.shape[0] - 1)].reshape(EEG.shape[1])
        Noise = ART[random.randint(0, ART.shape[0] - 1)].reshape(ART.shape[1])
        
        Noise = Noise * RMS(CleanEEG) / RMS(Noise) / SNR_Train[i]
        
        NoiseSet.append(Noise)
        TargetSet.append(CleanEEG)
        
    NoiseSet = np.array(NoiseSet)
    TargetSet = np.array(TargetSet)
    print('Noise Size:', NoiseSet.shape)
    print('Target Size:', TargetSet.shape)
    
    return TargetSet, NoiseSet

def dataGenerate(EEG, ART, Train_num, Val_num, Test_num, SNRmin, SNRmax):
    
    EEGnum = EEG.shape[0]
    ARTnum = ART.shape[0]
    print('Generating Train Set:')
    Train_input, Train_output = IOset(EEG[0:round(EEGnum * 0.8)], ART[0:round(ARTnum * 0.8)], Train_num, SNRmin, SNRmax)
    print('Generating Validation Set:')
    Val_input, Val_output = IOset(EEG[round(EEGnum * 0.8):round(EEGnum * 0.9)], ART[round(ARTnum * 0.8):round(ARTnum * 0.9)], Val_num, SNRmin, SNRmax)
    print('Generating Test Set:')
    Test_input, Test_output = IOset(EEG[round(EEGnum * 0.9):EEGnum], ART[round(ARTnum * 0.9):ARTnum], Test_num, SNRmin, SNRmax)
    
    return Train_input, Train_output, Val_input, Val_output, Test_input, Test_output

Train_target, Train_noise, Val_target, Val_noise, Test_target, Test_noise=dataGenerate(EEG, ART, Train_num, Val_num, Test_num, SNR_min, SNR_max)

##########################################

# Save Datasets

np.save(f'SC_pipeline/data/Train_{artifact}_target.npy', Train_target)
np.save(f'SC_pipeline/data/Train_{artifact}_noise.npy', Train_noise)
np.save(f'SC_pipeline/data/Val_{artifact}_target.npy', Val_target)
np.save(f'SC_pipeline/data/Val_{artifact}_noise.npy', Val_noise)
np.save(f'SC_pipeline/data/Test_{artifact}_target.npy', Test_target)
np.save(f'SC_pipeline/data/Test_{artifact}_noise.npy', Test_noise)

mkdir('SC_pipeline/data/input')
mkdir('SC_pipeline/data/output/csv')
mkdir('SC_pipeline/data/output/mat')