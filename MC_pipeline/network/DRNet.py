import torch
import torch.nn as nn

class DRNet(nn.Module):
    #Deep Residual Denoise net
    def __init__(self, in_channels = 60):
        super(DRNet, self).__init__()
        
        self.decomposer = BasicNet(In_Channels = in_channels)
        self.reconstructor = BasicNet(In_Channels = in_channels)
        
        self.decomposer_emb = nn.Conv1d(in_channels = 32, out_channels = in_channels, kernel_size = 1)
        self.reconstructor_squeeze = nn.Conv1d(in_channels = 32, out_channels = in_channels, kernel_size = 1)
        self.time_labeler_squeeze = nn.Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = 15, padding = 7)
        self.Sigmoid = nn.Sigmoid()
        
        #self.out_mix = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1)
        
    def forward(self, x):
        
        std = torch.std(x)
        x = torch.div(x, std)
        
        embedding_x = self.decomposer_emb(self.decomposer(x))
        
        out = self.reconstructor_squeeze(self.reconstructor(embedding_x))
        
        time_weight = self.Sigmoid(self.time_labeler_squeeze(embedding_x))
        out = torch.mul(x, time_weight) + torch.mul(out, 1 - time_weight)
        
        out = torch.mul(out, std)
        
        return out
        
    
class SE_InceptionBlock(nn.Module):
    
    '''
    input: [B, in_channels, T]
    output: [B, 32, T]
    '''
    
    def __init__(self, in_channels, SEreduction = 4):
        
        super(SE_InceptionBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 11, padding = 5)
        self.conv4 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 3, dilation = 7, padding = 7)

        # self.bn = nn.BatchNorm1d(num_features = 20, affine=True)
        
        assert 32 % SEreduction == 0, f'20%{SEreduction}!=0;'
        self.SEfc1 = nn.Linear(in_features = 32, out_features = 32 // SEreduction)
        self.SEfc2 = nn.Linear(in_features = 32 // SEreduction, out_features = 32)
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
    output: [B, 40, T]
    '''
    
    def __init__(self, in_channels):
        
        super(InceptionBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 11, padding = 5)
        self.conv4 = nn.Conv1d(in_channels = in_channels, out_channels = 8, kernel_size = 3, dilation = 7, padding = 7)
        self.reconv = nn.Conv1d(in_channels = 32, out_channels = in_channels, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        
        y_1 = self.conv1(x)
        y_2 = self.conv2(x)
        y_3 = self.conv3(x)
        y_4 = self.conv4(x)
        
        y = torch.cat((y_1, y_2, y_3, y_4), dim = 1)
        y = self.reconv(y)
        
        gelu = torch.nn.GELU()
        y = gelu(y)
        
        return y
    
class BasicNet(nn.Module):
    
    def __init__(self, In_Channels = 1):
        super(BasicNet, self).__init__()
        
        self.IB1 = InceptionBlock(in_channels = In_Channels)
        self.IB2 = InceptionBlock(in_channels = In_Channels)
        self.IB3 = InceptionBlock(in_channels = In_Channels)
        self.SEIB4 = SE_InceptionBlock(in_channels = In_Channels)
        
        #self.squeeze = nn.Conv1d(in_channels = 20, out_channels = 1, kernel_size = 1)
        
    def forward(self, x):
        
        x_1 = self.IB1(x)
        x_2 = self.IB2(x_1)
        x_3 = self.IB3(x_2)
        x_4 = self.SEIB4(x_3 + x_1)
        
        #y = self.squeeze(x_4)
        
        return x_4
