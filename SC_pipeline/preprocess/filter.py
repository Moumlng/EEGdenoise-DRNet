from scipy import signal
import numpy as np
import random
import matplotlib.pyplot as plt

## Data Freq:256Hz, EEG Low Pass: 50Hz
## Use Butterworth Lowpass at 50Hz

Fs = 256
b, a = signal.butter(N = 10, Wn = 50*2/Fs, btype = 'low', analog = False)

def prefilt(input, b = b, a = a, axis = 0):
    output = signal.filtfilt(b = b, a = a, x = input, axis = axis)
    return output

if __name__ == '__main__':
    
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency  
    
    t = np.linspace(0, 2, 512, False)
    #test_in = np.sin(2*np.pi*25*t) + np.sin(2*np.pi*100*t)
    test_in = np.load('SC_pipeline\data\EMG_all_epochs.npy')
    test_in = test_in[random.randint(0, test_in.shape[0]-1)]
    test_in = test_in / max(abs(test_in))
    test_out = prefilt(test_in)
    fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True)
    ax1.plot(t, test_in, color = 'red')
    ax1.axis([0, 2, -1, 1])
    ax1.set_title('Input EMG')
    ax2.plot(t, test_out, color = 'green')
    ax2.axis([0, 2, -1, 1])
    ax2.set_title('After Filter 80Hz Lowpass')
    
    plt.tight_layout()
    plt.show()
    