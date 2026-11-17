import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from transkconv import Convde
        

class Conv_block42(nn.Module):
    def __init__(self, filtersin, filters, imsize=None, block_nums=2, kernel_size=[3, 3]):
        super(Conv_block42, self).__init__()
        self.filt = filters
        
        self.conv0 = Convde(filtersin, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.conv0_ = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.batch0 = torch.nn.InstanceNorm2d(filters, track_running_stats=False, affine=False)
        self.batch0_ = torch.nn.InstanceNorm2d(filters, track_running_stats=False, affine=False)

        '''self.conv1 = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.conv1_ = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.batch1 = torch.nn.InstanceNorm2d(filters, track_running_stats=False)
        self.batch1_ = torch.nn.InstanceNorm2d(filters, track_running_stats=False)'''

        '''self.conv2 = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.conv2_ = Convde(filters, filters, kernel_size=[3, 3], padding='same', dilation=(1, 1))
        self.batch2 = torch.nn.InstanceNorm2d(filters, track_running_stats=False)
        self.batch2_ = torch.nn.InstanceNorm2d(filters, track_running_stats=False)'''

    def forward(self, inputs):
        x0 = inputs
        #x0 = F.pad(x0, (0, 1, 0, 1))
        x0 = self.conv0(x0)
        x0 = self.batch0(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')
        #x0 = F.pad(x0, (1, 0, 1, 0))
        x0 = self.conv0_(x0)
        x0 = self.batch0_(x0)
        
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')

        #x0 = F.pad(x0, (0, 2, 0, 2))
        '''x0 = self.conv1(x0)
        x0 = self.batch1(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')
        #x0 = F.pad(x0, (0, 2, 0, 2))
        x0 = self.conv1_(x0)
        x0 = self.batch1_(x0)
        x0 = torch.nn.functional.relu(x0)'''

        '''x0 = self.conv2(x0)
        x0 = self.batch2(x0)
        x0 = torch.nn.functional.relu(x0)
        #x0 = F.interpolate(x0, size=[inputs.shape[-2], inputs.shape[-1]], mode='bilinear')
        #x0 = F.pad(x0, (0, 2, 0, 2))
        x0 = self.conv2_(x0)
        x0 = self.batch2_(x0)
        x0 = torch.nn.functional.relu(x0)'''
        return x0

class Unet_Encode(nn.Module):
    def __init__(self, filtersin, filters):
        super(Unet_Encode, self).__init__()
        self.conv0 = Conv_block42(filtersin, filters, imsize=512, block_nums=2)
        self.pool0 = nn.MaxPool2d(kernel_size=[2, 2], stride=2, padding=(0,0))
        self.conv1 = Conv_block42(filters, filters * 2, imsize=256, block_nums=2)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2, padding=(0,0))
        self.conv2 = Conv_block42(filters * 2, filters * 4, imsize=128, block_nums=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2, padding=(0,0))
        self.conv3 = Conv_block42(filters * 4, filters * 8, imsize=64, block_nums=2)

    def forward(self, inputs):
        x0 = inputs
        x0 = self.conv0(x0)
        
        x1 = 1 * x0
        x0 = self.pool0(x0)
        x0 = self.conv1(x0)
        x2 = 1 * x0
        x0 = self.pool1(x0)
        x0 = self.conv2(x0)
        x3 = 1 * x0
        x0 = self.pool2(x0)
        x0 = self.conv3(x0)
        x4 = 1*x0 
        return x1, x2, x3, x4


class Unet_Decode(nn.Module):
    def __init__(self, filters):
        super(Unet_Decode, self).__init__()
        self.convu1 = Conv_block42(filters * 4+filters * 8, filters * 4,imsize=128, block_nums=2)
        
        self.convu2 = Conv_block42(filters * 2+filters * 4, filters * 2,imsize=256, block_nums=2)
        '''self.convu2 = Conv_block42(filters * 4, filters * 2,imsize=256, block_nums=2)
        self.conv2 = nn.Conv2d(filters * 2, filters * 4, kernel_size=[1, 1], padding='same', dilation=(1, 1))'''
        #self.batch2 = torch.nn.InstanceNorm2d(filters * 4, track_running_stats=False)
        
        
        self.convu3 = Conv_block42(filters * 1+filters * 2, filters,imsize=512, block_nums=2)
        '''self.convu3 = Conv_block42(filters * 2, filters,imsize=512, block_nums=2)
        self.conv3 = nn.Conv2d(filters * 1, filters * 2, kernel_size=[1, 1], padding='same', dilation=(1, 1))'''

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        x0 = F.interpolate(x4, size=[x4.shape[-2]*2, x4.shape[-1]*2], mode='bilinear')
        x0 = torch.cat([x0, x3], dim=1)
        x3 = self.convu1(x0)
        x0 = F.interpolate(x3, size=[x3.shape[-2]*2, x3.shape[-1]*2], mode='bilinear')
        
        x0 = torch.cat([x0, x2], dim=1)
        '''x2 = self.conv2(x2)
        x0 = x0*(F.sigmoid(x2))'''
        x22 = self.convu2(x0)
        x0 = F.interpolate(x22, size=[x22.shape[-2]*2, x22.shape[-1]*2], mode='bilinear')
        x0 = torch.cat([x0, x1], dim=1)
        '''x1 = self.conv3(x1)
        x0 = x0*(F.sigmoid(x1))'''
        outputs = self.convu3(x0)

        return outputs



class ConnectUnet(nn.Module):
    def __init__(self,unetn=3, base_filters=32):
        super(ConnectUnet, self).__init__()
        self.unetn=unetn
        filters=base_filters
        self.unet_decode0 = Unet_Decode(filters)
        self.unet_encode0 = Unet_Encode(3, filters)
        self.conv_out0 = nn.Conv2d(filters, 1, kernel_size=[1, 1], padding='same')
        
        if unetn>1:
            self.unet_decode1 = Unet_Decode(filters)
            self.unet_encode1 = Unet_Encode(filters+3, filters)
            self.conv_out1 = nn.Conv2d(filters, 1, kernel_size=[1, 1], padding='same')
        if unetn>2:
            self.unet_decode2 = Unet_Decode(filters)
            self.unet_encode2 = Unet_Encode(filters+3, filters)
            self.conv_out2 = nn.Conv2d(filters, 1, kernel_size=[1, 1], padding='same')
    def forward(self, inputs):
        inputs = inputs
        x0 = self.unet_encode0(inputs)
        x0 = self.unet_decode0(x0)
        output_seg0 = self.conv_out0(x0)
        output_seg0 = torch.sigmoid(output_seg0)
        if self.unetn>1:
            x0 = torch.cat([x0,inputs],1)
            x0 = self.unet_encode1(x0)
            x0= self.unet_decode1(x0)
            output_seg1 = self.conv_out1(x0)
            output_seg1 = torch.sigmoid(output_seg1)
        if self.unetn>2:
            x0 = torch.cat([x0,inputs],1)
            x0 = self.unet_encode2(x0)
            x0 = self.unet_decode2(x0)
            output_seg2 = self.conv_out2(x0)
            output_seg2 = torch.sigmoid(output_seg2)
        if self.unetn==3:
            return [output_seg0, output_seg1, output_seg2], x0
        elif self.unetn==2:
            return [output_seg0, output_seg1], x0
        elif self.unetn==1:
            return [output_seg0], x0
