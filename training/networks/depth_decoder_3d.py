

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers_3d import *


class DepthDecoder_3d(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_3d, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        #self.num_ch_enc = np.array([64,64,128,256,512])
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
            #if self.use_skips and i > 0:
                #num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3_2d(self.num_ch_dec[s], self.num_output_channels)
 
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        #x=torch.mean(x,x.ndimension()-3)
        #print("yes",input_features[0].size())
        #print("s",input_features[1].size())
        #print("s",input_features[2].size())
        #print("s",input_features[3].size())
        #print("s",input_features[4].size())
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x=x[:,:,:1,:,:]
            x = [upsample(x)]
            #print(i)
            #print("n",x[0].size())
            #if self.use_skips and i > 0:
                #x +=[input_features[i - 1]]
            x = torch.cat((x), 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                #x1=x[:,:,0,:,:]
                #x1=torch.squeeze(x1,dim=2)
                #print(x1.size())
                x1=torch.mean(x,x.ndimension()-3)
                #x1=torch.min(x,x.ndimension()-3)
                #print("new",x.size())
                self.outputs[("disp", i)] = self.sigmoid((self.convs[("dispconv", i)](x1)))
                #self.outputs[("disp", i)] = torch.mean(self.outputs[("disp", i)],self.outputs[("disp", i)].ndimension()-3)
                
        return self.outputs
