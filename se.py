import torch
import torch.nn as nn
import math


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        squeeze_tensor = input_tensor.mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10], subband_num=1):

        super(ChannelTimeSenseSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels // subband_num), 
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels // subband_num), 
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels // subband_num), 
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2) 
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0] 

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSEWeightLayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        
        super(ChannelTimeSenseSEWeightLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),  
            nn.AdaptiveAvgPool1d(1), 
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels), 
            nn.AdaptiveAvgPool1d(1), 
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels), 
            nn.AdaptiveAvgPool1d(1), 
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2) 
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0] 

       
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor, fc_out_2.view(a, b, 1)


class ChannelDeepTimeSenseSELayer(nn.Module):
    

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        
        super(ChannelDeepTimeSenseSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),  
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels), 
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1) 

        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]  

        
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class Conv_Attention_Block(nn.Module):
    def __init__(
            self,
            num_channels,
            kersize=[3, 5, 10]
    ):
        
        super().__init__()
        self.conv1d = nn.Conv1d(num_channels, num_channels, kernel_size=kersize, groups=num_channels)
        self.attention = SelfAttentionlayer(amp_dim=num_channels, att_dim=num_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.active_funtion = nn.ReLU(inplace=True)

    def forward(self, input):
        input = (self.conv1d(input)).permute(0, 2, 1)  
        input = self.attention(input, input, input)  
        output = self.active_funtion(self.avgpool(input.permute(0, 2, 1)))  
        return output


class ChannelTimeSenseAttentionSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        
        super(ChannelTimeSenseAttentionSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.smallConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[0])
        self.middleConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[1])
        self.largeConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[2])

        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)  
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0] 

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelCBAMLayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        
        super(ChannelCBAMLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        
        mean_squeeze_tensor = input_tensor.mean(dim=2)
        max_squeeze_tensor, _ = torch.max(input_tensor, dim=2)  
        mean_fc_out_1 = self.relu(self.fc1(mean_squeeze_tensor))
        max_fc_out_1 = self.relu(self.fc1(max_squeeze_tensor))
        fc_out_1 = mean_fc_out_1 + max_fc_out_1
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = mean_squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelECAlayer(nn.Module):

    def __init__(self, channel, k_size=3):
        super(ChannelECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        y = self.avg_pool(x)

        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SelfAttentionlayer(nn.Module):
    
    def __init__(self, amp_dim=257, att_dim=257):
        super(SelfAttentionlayer, self).__init__()
        self.d_k = amp_dim
        self.q_linear = nn.Linear(amp_dim, att_dim)
        self.k_linear = nn.Linear(amp_dim, att_dim)
        self.v_linear = nn.Linear(amp_dim, att_dim)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(att_dim, amp_dim)

    def forward(self, q, k, v):
        q = self.q_linear(q) 
        k = self.k_linear(k)
        v = self.v_linear(v)
        output = self.attention(q, k, v)
        output = self.out(output)
        return output  

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.sigmoid(scores)
        output = torch.matmul(scores, v)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x.clone()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x) 
        output_tensor = torch.mul(input_tensor, x)
        return output_tensor
    
if __name__ == "__main__":
    data=torch.rand(32,1024, 5)
    ca=ChannelSELayer(1024)
    out = ca(data)
    print(out.shape)