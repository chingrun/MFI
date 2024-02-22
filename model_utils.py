import torch
import torch.nn as nn
import numpy as np
from se import ChannelSELayer
import pdb

device = 'cuda:0'
    
class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias,padding=padding),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)
def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)
    

class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)
    

class Block(nn.Module):
    def __init__(self, dim, dim_out, innerchannel) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU1D(innerchannel, dim, 1),
                                   ConvBNReLURes1D(dim, 1, 1, True),
                                   ConvBNReLU1D(dim, dim_out, 1)
                                   )
    def forward(self, x):
        return self.block(x)

def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    in_channels = 0
    for scale in feat_scales:
        in_channels = inner_channel*channel_multiplier[scale]
    return in_channels

class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU1D(dim, dim_out, 1),
            ConvBNReLURes1D(dim_out, 1, 1, True),
            ChannelSELayer(num_channels=dim_out, reduction_ratio=2),
            ConvBNReLURes1D(dim_out, 1, 1, True),
            ConvBNReLU1D(dim_out, dim_out, 1)
            
        )

    def forward(self, x):
        return self.block(x)


class outModule(nn.Module):
    def __init__(self, feat_scales, out_channels=6, inner_channel=None, channel_multiplier=None, time_steps=None) -> None:
        super().__init__()
        feat_scales.sort(reverse=False)
        self.feat_scales    = feat_scales
        self.time_steps = time_steps
        self.decoder = nn.ModuleList()  #len=9
        dim_out = None
        
        for i in range(0, len(self.feat_scales)):
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
            self.decoder.append(
                Block(dim=dim, dim_out=dim, innerchannel=inner_channel)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )
        if dim_out == None:
            dim_out = dim

        clfr_emb_dim = int(dim_out/64)
        self.clfr_stg1 = ConvBNReLU1D(dim_out, int(dim_out/2),3, padding=1)
        self.clfr_stg2 = ConvBNReLU1D(int(dim_out/8), int(dim_out/16),3, padding=1)
        self.average_pool = nn.AvgPool1d(4,4)
        self.clfr_stgn = ConvBNReLU1D(clfr_emb_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, f_A):
        # Decoder
        lvl=0
        for layer in self.decoder:
            if isinstance(layer, Block):

                diff = layer(f_A)
                
                if lvl!=0:

                    diff = diff + x
                lvl+=1
            else:
                diff = layer(diff)
                x = diff
        x = diff if lvl<2 else x

        cm = self.clfr_stg1(x)
        cm = self.average_pool(cm.permute(0,2,1))
        cm = self.clfr_stg2(cm.permute(0,2,1))
        cm = self.average_pool(cm.permute(0,2,1))
        cm = self.clfr_stgn(cm.permute(0,2,1))

        return cm
    
if __name__ == "__main__":

    feat_scales = [0,1,2]
    time_steps = [0,2,4]
    channel_multiplier = [1,1,1]
    timedata = torch.rand(32, 512, 1024).cuda()
    model = outModule(feat_scales, 768, inner_channel=512, channel_multiplier=channel_multiplier, time_steps=time_steps).cuda()
    out = model(timedata)
    print(out.shape)