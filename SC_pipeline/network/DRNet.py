import torch
import torch.nn as nn

class DRNet(nn.Module):
    def __init__(self):
        super(DRNet, self).__init__()
        
        self.decomposer = BasicNet(In_Channels = 1)
        self.reconstructor = BasicNet(In_Channels = 6)
        
        self.decomposer_emb = nn.Conv1d(in_channels = 20, out_channels = 6, kernel_size = 1)
        self.reconstructor_squeeze = nn.Conv1d(in_channels = 20, out_channels = 1, kernel_size = 1)
        self.time_labeler_squeeze = nn.Conv1d(in_channels = 6, out_channels = 1, kernel_size = 15, padding = 7)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        std = torch.std(x)
        x = x / std
        
        x = torch.unsqueeze(x, dim = 1)
        
        embedding_x = self.decomposer_emb(self.decomposer(x))
        
        out = self.reconstructor_squeeze(self.reconstructor(embedding_x))
        time_weight = self.Sigmoid(self.time_labeler_squeeze(embedding_x))
        out = torch.mul(x, time_weight) + torch.mul(out, 1 - time_weight)
        
        out = torch.squeeze(out, dim = 1)
        
        out = out * std
        
        return out
        
    
class SE_InceptionBlock(nn.Module):
    
    '''
    input: [B, in_channels, T]
    output: [B, 20, T]
    '''
    
    def __init__(self, in_channels, SEreduction = 4):
        
        super(SE_InceptionBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 11, padding = 5)
        self.conv4 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 3, dilation = 7, padding = 7)
        
        # self.bn = nn.BatchNorm1d(num_features = 20, affine=True)
        
        assert 20 % SEreduction == 0, f'20%{SEreduction}!=0;'
        self.SEfc1 = nn.Linear(in_features = 20, out_features = 20 // SEreduction)
        self.SEfc2 = nn.Linear(in_features = 20 // SEreduction, out_features = 20)
        self.SErelu = nn.ReLU(inplace = True)
        self.SEsigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        y_1 = self.conv1(x)
        y_2 = self.conv2(x)
        y_3 = self.conv3(x)
        y_4 = self.conv4(x)
        
        y = torch.cat((y_1, y_2, y_3, y_4), dim = 1)
        
        gelu = torch.nn.GELU()
        y = gelu(y)
        # y = self.bn(y)
        
        out = y.mean(dim = 2)
        out = self.SErelu(self.SEfc1(out))
        out = self.SEsigmoid(self.SEfc2(out))
        
        return y * out.unsqueeze(2)

class InceptionBlock(nn.Module):
    
    '''
    input: [B, in_channels, T]
    output: [B, 20, T]
    '''
    
    def __init__(self, in_channels):
        
        super(InceptionBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 11, padding = 5)
        self.conv4 = nn.Conv1d(in_channels = in_channels, out_channels = 5, kernel_size = 3, dilation = 7, padding = 7)
        
    def forward(self, x):
        
        y_1 = self.conv1(x)
        y_2 = self.conv2(x)
        y_3 = self.conv3(x)
        y_4 = self.conv4(x)
        
        y = torch.cat((y_1, y_2, y_3, y_4), dim = 1)
        
        gelu = torch.nn.GELU()
        y = gelu(y)
        
        return y
    
class BasicNet(nn.Module):
    
    def __init__(self, In_Channels = 1):
        super(BasicNet, self).__init__()
        
        self.IB1 = InceptionBlock(in_channels = In_Channels)
        self.IB2 = InceptionBlock(in_channels = 20)
        self.IB3 = InceptionBlock(in_channels = 20)
        self.SEIB4 = SE_InceptionBlock(in_channels = 20)
        
        #self.squeeze = nn.Conv1d(in_channels = 20, out_channels = 1, kernel_size = 1)
        
    def forward(self, x):
        
        x_1 = self.IB1(x)
        x_2 = self.IB2(x_1)
        x_3 = self.IB3(x_2)
        x_4 = self.SEIB4(x_3 + x_1)
        
        #y = self.squeeze(x_4)
        
        return x_4
