import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import outModule, ConvBNReLU1D, ConvBNReLURes1D
from se import ChannelSELayer, SpatialAttention
import pdb

class Model(nn.Module):
    def __init__(self, device, inchannel, outNum, feat_scales, feature_outchannel, num_points, time_steps, channel_multiplier) -> None:
        super().__init__()
        self.device = device
        self.outNum = outNum
        self.feat_scales = feat_scales
        self.channel_multiplier = channel_multiplier
        self.feature_outchannel = feature_outchannel
        self.time_steps = time_steps
        self.num_points = num_points
        self.feature_length = len(feature_outchannel)
        outchannel = 0
        for j in range(self.feature_length):
            outchannel += self.feature_outchannel[j]

        convinblks = []
        convinblks.append(ConvBNReLU1D(inchannel, int(outchannel/4), 1))
        convinblks.append(ConvBNReLURes1D(int(outchannel/4)))

        for i in range(1, 4):
            convinblks.append(ConvBNReLU1D(i*int(outchannel/4), (i+1)*int(outchannel/4), 1))
            convinblks.append(ConvBNReLURes1D((i+1)*int(outchannel/4)))

        self.convin = nn.Sequential(*convinblks)

        self.conv2 = ConvBNReLU1D(128, 64,1)
        self.conv3 = nn.Conv1d(64, self.outNum, 1)
        
        self.outMod = outModule(self.feat_scales, 128, inner_channel=240, channel_multiplier=self.channel_multiplier, time_steps=self.time_steps).to(self.device)

        self.sa1 = SpatialAttention()
        self.ca1 = ChannelSELayer(inchannel-3)
        self.psaconv = nn.Sequential(ConvBNReLU1D(inchannel+3, outchannel),
                                     ConvBNReLURes1D(outchannel),
                                     ConvBNReLURes1D(outchannel),
                                     ConvBNReLU1D(outchannel, outchannel))
        self.sa2 = ConvBNReLU1D(3, outchannel)
        self.ca2 = ConvBNReLU1D(inchannel-3, outchannel)
        self.psa2 = ConvBNReLU1D(outchannel, outchannel)
        self.noatt = ConvBNReLU1D(inchannel, outchannel)

        self.ap = nn.AvgPool1d(4, 4)

        self.featureExtract = nn.Sequential(ConvBNReLU1D(180, 128),##lithology_oriented
                                            ConvBNReLU1D(128,64),##lithology_oriented
                                            ConvBNReLU1D(64,32),##lithology_oriented
                                            ConvBNReLU1D(32, self.outNum))##lithology_oriented
                                            
    def forward(self, x):
        x = x.float()
        batch = x.shape[0]        
        sa1 = self.sa1(x[:,:,0:3]).permute(0,2,1)
        ca1 = self.ca1(x[:,:,3:].permute(0,2,1))
        psa1 = torch.cat([x.transpose(2,1), sa1], dim=1)
        psa1 = self.psaconv(psa1)

        sac = self.sa2(sa1)
        cac = self.ca2(ca1)
        psac = self.psa2(psa1)
        noatt = self.noatt(x.transpose(2,1))       

        apsa = self.ap(sac.permute(0,2,1)).permute(0,2,1)
        apca = self.ap(cac.permute(0,2,1)).permute(0,2,1)
        appsa = self.ap(psac.permute(0,2,1)).permute(0,2,1)
        apnoatt = self.ap(noatt.permute(0,2,1)).permute(0,2,1)

        ft = torch.cat([apsa, apca, appsa, apnoatt], dim=1)
        with torch.no_grad():                                                       ##lithology_oriented
            # ftExtract = self.featureExtract(torch.cat([apsa, apca, appsa], dim=1))  ##lithology_oriented

            apsaout = torch.var(apsa, dim=1).unsqueeze(1)
            apcaout = torch.var(apca, dim=1).unsqueeze(1)
            appsaout = torch.var(appsa, dim=1).unsqueeze(1)
            apnoattout = torch.var(apnoatt, dim=1).unsqueeze(1)
            ftout = torch.cat([apsaout, apcaout, appsaout, apnoattout], dim=1)
            # ftout = torch.cat([apsa[:,0,:], apca[:,0,:], appsa[:,0,:], apnoatt[:,0,:]], dim=1)
        
        
        o = self.outMod(ft)
        
        o = self.conv2(o)
        o = self.conv3(o)

        logits = o.transpose(2,1).contiguous()
        logits = F.log_softmax(logits.view(-1, self.outNum), dim=-1)

        logits = logits.view(batch, self.num_points, self.outNum)
        # return logits, ftExtract ##lithology_oriented
        # return logits, ftout ##infer usage
        return logits

if __name__ == "__main__":
    data = torch.rand(8, 1024, 150)
    feat_scales = [0,1,2,3]
    feature_outchannel = [16, 32,64,128]
    feature_kernels = [32,32,32,32]
    time_steps = [0,0,0,0]
    channel_multiplier = [1,2,4,8]
    model = Model('cuda:0',150, feat_scales, feature_outchannel, feature_kernels, time_steps, channel_multiplier).to('cuda:0')
    out = model(data.to('cuda:0'))
    print(out.shape)
