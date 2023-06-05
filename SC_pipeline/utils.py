import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import random
import time

def RMS(data):
    return np.sqrt(np.sum(data**2)/data.shape[0])

def SNR(signal, noise):
    return 20 * np.log10(RMS(signal) / RMS(noise))

def RMSE(result, target):
    assert result.shape == target.shape
    return RMS(result - target)

def RRMSE(result, target):
    assert result.shape == target.shape
    return RMS(result - target) / RMS(target)

def standardize(data):
    return (data - np.mean(data)) / np.std(data)

def train_val_plot(train_losses, val_losses, save_loc = 'models/loss.jpg'):
    fig, ax = plt.subplots()
    
    x = np.linspace(0,len(train_losses)-1, len(train_losses))
    ax.plot(x, train_losses, label = 'Train Loss')
    ax.plot(x, val_losses, label = 'Val Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Train process. Final loss = {val_losses[-1]}')
    ax.legend()
    
    plt.savefig(save_loc, dpi = 150)
    
def mkdir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'New Folder: "{path}"')
    else:
        print(f'Folder "{path}" Exists.')
        
def chooseIndex(length):
    random.seed(time.time())
    id = random.randint(0, length-1)
    return id    
        
def plot_valLoader_set(raw, target, output, index = 0):
    
    plt.figure(figsize=(12,8), dpi = 100)
    x = np.linspace(0,2,512)
    
    ax = plt.subplot(4,1,1)
    plt.plot(x, raw.cpu().detach().numpy()[index], label = 'raw', color = '#FF6600')
    plt.plot(x, target.cpu().detach().numpy()[index], label = 'target', color = '#339966')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    #ax.set_title(f'Input SNR: {SNR(target.cpu().detach().numpy()[index], (raw.cpu().detach().numpy()[index] - target.cpu().detach().numpy()[index]))}')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(4,1,2)
    plt.plot(x, raw.cpu().detach().numpy()[index], label = 'raw', color = '#FF6600')
    plt.plot(x, output.cpu().detach().numpy()[index], label = 'output', color = '#003366')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(4,1,3)
    plt.plot(x, target.cpu().detach().numpy()[index], label = 'target', color = '#339966')
    plt.plot(x, output.cpu().detach().numpy()[index], label = 'output', color = '#003366')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')    
    #ax.set_title(f'Output SNR: {SNR(target.cpu().detach().numpy()[index], (output.cpu().detach().numpy()[index] - target.cpu().detach().numpy()[index]))}')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(4,1,4)
    plt.plot(x, output.cpu().detach().numpy()[index], label = 'output', color = '#003366')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    plt.legend(loc = 'upper right')
    
    plt.tight_layout()
    
def plot_numpy_set(raw, target, output, index, prob = None, fs = 256):
    
    dots = raw.shape[0]
    plt.figure(figsize=(12,6), dpi = 100)
    x = np.linspace(0,dots/fs,dots)
    
    ax = plt.subplot(3,1,1)
    plt.plot(x, raw, label = 'raw', color = '#FF6600')
    plt.plot(x, target, label = 'target', color = '#339966')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    #ax.set_title(f'Input SNR = {SNR(target, (raw - target))}')
    ax.set_title(f'Index = {index}', loc = 'left')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(3,1,2)
    plt.plot(x, raw, label = 'raw', color = '#FF6600')
    plt.plot(x, output, label = 'output', color = '#003366')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(3,1,3)
    plt.plot(x, target, label = 'target', color = '#339966')
    plt.plot(x, output, label = 'output', color = '#003366')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    #ax.set_title(f'Output SNR = {SNR(target , (output - target))}')
    plt.legend(loc = 'upper right')
    '''
    ax = plt.subplot(4,1,4)
    plt.plot(x, output, label = 'output', color = '#003366')
    if prob is not None:
        plt.plot(x, prob, label = 'prob', color = 'red')
    
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    plt.legend(loc = 'upper right')
    '''
    plt.tight_layout()
    
def plt_spec_set(input, target, output, Fs = 256, NFFT = 64, noverlap = 63):
    
    plt.figure()
    
    plt.subplot(3,1,1)
    plt.specgram(input, Fs = Fs, NFFT = NFFT, noverlap = noverlap)
    plt.title('Raw')
    plt.subplot(3,1,2)
    plt.specgram(target, Fs = Fs, NFFT = NFFT, noverlap = noverlap)
    plt.title('Target')
    plt.subplot(3,1,3)
    plt.specgram(output, Fs = Fs, NFFT = NFFT, noverlap = noverlap)
    plt.title('output')
    
    plt.tight_layout()
    
def pipeline_plot(raw, target, output1, output2, index, prob, threshold, fs = 256):
    
    plt.figure(figsize=(12,9), dpi = 100)
    dots = raw.shape[0]
    x = np.linspace(0, dots/fs, dots)
    th = threshold * np.ones_like(raw)
    
    ax = plt.subplot(5,1,1)
    plt.plot(x, raw, label = 'raw', color = '#FF6600')
    plt.plot(x, target, label = 'target', color = '#339966')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    ax.set_title(f'Input SNR = {SNR(target, (raw - target))}')
    ax.set_title(f'Index = {index}', loc = 'left')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(5,1,2)
    plt.plot(x, output1, label = 'output_1', color = '#003399')
    plt.plot(x, target, label = 'target', color = '#339966')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(5,1,3)
    plt.plot(x, prob, label = 'prob', color = 'green', linestyle = '-.')
    plt.plot(x, th, label = 'threshold', color = 'red', linestyle = '--')
    ax.set_xlabel('time/s')
    ax.set_ylabel('Probablity')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(5,1,4)
    plt.plot(x, target, label = 'target', color = '#339966')
    plt.plot(x, output2, label = 'output_2', color = '#003366')
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    ax.set_title(f'Output SNR = {SNR(target , (output2 - target))}')
    plt.legend(loc = 'upper right')
    
    ax = plt.subplot(5,1,5)
    plt.plot(x, output2, label = 'output_2', color = '#003366')
    '''
    if prob is not None:
        plt.plot(x, prob, label = 'prob', color = 'red')
    '''
    ax.set_xlabel('time/s')
    ax.set_ylabel('amp')
    plt.legend(loc = 'upper right')
    
    plt.tight_layout()
    
def stft_plt(x, title, fs = 256, t = 2):
    f, t, zxx = signal.stft(x, fs = fs, nperseg = 256, noverlap = 255)
    flen = f.shape[0]
    f = f[0: round(flen * 50/128)]
    zxx = zxx[0: round(flen * 50/128), :]
    plt.figure(figsize = (12,6))
    plt.subplot(2,1,1)
    plt.title(title)
    plt.plot(t[:-1], x)
    plt.xlabel('time/s')
    plt.ylabel('standardized amp')
    plt.subplot(2,1,2)
    plt.pcolormesh(t, f, abs(zxx), shading = 'auto', cmap = 'Reds', vmin = 0, vmax = 0.4)
    #plt.colorbar()
    plt.xlabel('time/s')
    plt.ylabel('freq/Hz')
    
    plt.tight_layout()

if __name__ == '__main__':
    
    EEG = np.load('pipeline/data/EEG_all_epochs.npy')
    Index = chooseIndex(EEG.shape[0])
    EEG = EEG[Index]
    EEG = EEG/np.std(EEG)
    stft_plt(EEG, 'EEG')
    
    EMG = np.load('pipeline/data/EMG_all_epochs.npy')
    Index = chooseIndex(EMG.shape[0])
    EMG = EMG[Index]
    EMG = EMG/np.std(EMG)
    stft_plt(EMG, 'EMG')
    
    EOG = np.load('pipeline/data/EOG_all_epochs.npy')
    Index = chooseIndex(EOG.shape[0])
    EOG = EOG[Index]
    EOG = EOG/np.std(EOG)
    stft_plt(EOG, 'EOG')
    
    plt.show()