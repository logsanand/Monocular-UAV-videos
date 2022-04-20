
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc=np.array([64, 64, 128, 256, 512]), scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        output=[]
        # decoder
        x1 = input_features
        print(x1[1].shape)
        
        for i in range(3, -1, -1):
            #x = self.convs[("upconv", i, 0)](x)
            x=torch.FloatTensor(x1[i])
            print(x.size())
            if i==3:
               x=[upsample(x,1)]
            if i==2:
               x=[upsample(x,1)]
            if i==1:
               x=[upsample(x,4)]
            if i==0:
               x=[upsample(x,2)]

            #x = [upsample(x)]
            print(x[0].shape)
            #if self.use_skips and i > 0:
                #if i==4:
                    #x += [input_features[i - 1]]
                #else:
                    #x += [input_features[2]]
            #x = torch.cat(x, 1)
            #x = self.convs[("upconv", i, 1)](x)
            output.append(x[0])
        output1=torch.FloatTensor(output[1])
        output1.unsqueeze_(0)
        output2=torch.FloatTensor(output[2])
        output2.unsqueeze_(0)
        output3=torch.FloatTensor(output[3])
        output3.unsqueeze_(0)
        print(output[1].shape)
        outputs=torch.cat((output1,output2,output3),0)
        outputs=outputs.permute(1,0,2,3,4)
        print(outputs.size())

        return self.outputs


               
