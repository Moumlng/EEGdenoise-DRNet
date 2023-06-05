import torch.utils.data as data
import torch
import scipy.io as io
import numpy as np

class EEGdataset(data.Dataset):
    def __init__(self, nums_file = 292, data_dir = 'EC_train/', bias = 0):
        super(EEGdataset).__init__()
        self.clean_data = torch.tensor([])
        self.raw_data = torch.tensor([])
        for i in range(nums_file):
            
            idx = i + 1 + bias
            
            clean_mat = io.loadmat(data_dir + f'sub{idx}_clean.mat')
            clean = clean_mat['EEG_clean']
            clean = torch.from_numpy(clean)
            clean = torch.permute(clean, (2,0,1)).float()
            self.clean_data = torch.concat((self.clean_data, clean), dim = 0)
            
            raw_mat = io.loadmat(data_dir + f'sub{idx}_raw.mat')
            raw = raw_mat['EEG_raw']
            raw = torch.from_numpy(raw)
            raw = torch.permute(raw, (2,0,1)).float()
            self.raw_data = torch.concat((self.raw_data, raw), dim = 0)
            
        print('Clean data shape:', self.clean_data.shape)
        print('Raw data shape:', self.raw_data.shape)
        
    def __getitem__(self, index):
        
        clean = self.clean_data[index]
        raw = self.raw_data[index]
        
        return clean, raw
    
    def __len__(self):
        
        assert self.raw_data.shape == self.clean_data.shape
        return self.raw_data.shape[0]