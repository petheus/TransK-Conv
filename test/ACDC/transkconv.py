import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


'''class AddCoords(nn.Module):
    def __init__(self, imsize, with_r=True, xoy=2):
        super(AddCoords, self).__init__()
        self.xoy = xoy
        self.with_r = with_r
        self.x_dim = imsize
        self.y_dim = imsize
        xx_channel = torch.arange(self.y_dim, dtype=torch.float32)
        yy_channel = torch.arange(self.x_dim, dtype=torch.float32)
        self.register_buffer('xx_channel', xx_channel)
        self.register_buffer('yy_channel', yy_channel)
            

    def forward(self, input_tensor):
        self.batch_size_tensor = 1#input_tensor.shape[0]
        
        xcs = torch.sin(self.xx_channel / torch.pow(10000., self.xx_channel // 2 * 2 / self.y_dim))
        xcc = torch.cos(self.xx_channel / torch.pow(10000., self.xx_channel // 2 * 2 / self.y_dim))
        xx_channel = torch.where(self.xx_channel % 2 == 0, xcs, xcc)
        xx_channel = xx_channel.unsqueeze(0)
        xx_channel = xx_channel.unsqueeze(0)  # shape [1,1,y_dim]
        xx_channel = xx_channel.repeat(self.batch_size_tensor, self.x_dim, 1)
        xx_channel = xx_channel.unsqueeze(1)

       
        ycs = torch.sin(self.yy_channel / torch.pow(10000., self.yy_channel // 2 * 2 / self.x_dim))
        ycc = torch.cos(self.yy_channel / torch.pow(10000., self.yy_channel // 2 * 2 / self.x_dim))
        yy_channel = torch.where(self.yy_channel % 2 == 0, ycs, ycc)
        yy_channel = yy_channel.unsqueeze(0)
        yy_channel = yy_channel.unsqueeze(-1)  # shape [1,x_dim, 1]
        yy_channel = yy_channel.repeat(self.batch_size_tensor, 1, self.y_dim)
        yy_channel = yy_channel.unsqueeze(1)

        ret = input_tensor+xx_channel+yy_channel
        #ret = torch.cat([input_tensor, xx_channel, yy_channel], axis=1)

        if self.with_r:
            rr = torch.sqrt(torch.square(xx_channel) + torch.square(yy_channel))
            ret = ret+rr
            #ret = torch.cat([ret, rr], axis=1)

        return ret'''


class AddCoords(nn.Module):
    def __init__(self, with_r=True, xoy=2):
        super(AddCoords, self).__init__()
        self.xoy = xoy
        self.with_r = with_r

    def forward(self, input_tensor):
            
            
        x_dim = input_tensor.shape[2]
        y_dim = input_tensor.shape[3]
        xx_channel = torch.arange(y_dim, dtype=torch.float32).cuda()
        yy_channel = torch.arange(x_dim, dtype=torch.float32).cuda()
        batch_size_tensor = input_tensor.shape[0]
        xcs = torch.sin(xx_channel / torch.pow(10000., xx_channel // 2 * 2 / y_dim))
        xcc = torch.cos(xx_channel / torch.pow(10000., xx_channel // 2 * 2 / y_dim))
        xx_channel = torch.where(xx_channel % 2 == 0, xcs, xcc)
        xx_channel = xx_channel.unsqueeze(0)
        xx_channel = xx_channel.unsqueeze(0)  # shape [1,1,y_dim]
        xx_channel = xx_channel.repeat(batch_size_tensor, x_dim, 1)
        xx_channel = xx_channel.unsqueeze(1)

       
        ycs = torch.sin(yy_channel / torch.pow(10000., yy_channel // 2 * 2 / x_dim))
        ycc = torch.cos(yy_channel / torch.pow(10000., yy_channel // 2 * 2 / x_dim))
        yy_channel = torch.where(yy_channel % 2 == 0, ycs, ycc)
        yy_channel = yy_channel.unsqueeze(0)
        yy_channel = yy_channel.unsqueeze(-1)  # shape [1,x_dim, 1]
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, y_dim)
        yy_channel = yy_channel.unsqueeze(1)

        #ret = input_tensor+xx_channel+yy_channel
        ret = torch.cat([input_tensor, xx_channel, yy_channel], axis=1)

        if self.with_r:
            rr = torch.sqrt(torch.square(xx_channel) + torch.square(yy_channel))
            #ret = ret+rr
            ret = torch.cat([ret, rr], axis=1)

        return ret

class AttentionBlock(nn.Module):

    def __init__(self, filters, heads):
        super(AttentionBlock, self).__init__()
        self.filters = filters
        self.heads = heads

    def forward(self, inputs):
        qw, kw, vw = inputs
        input_shape0 = qw.shape
        dk0 = math.sqrt(self.filters)
        en_q = qw
        en_q = torch.reshape(en_q, [input_shape0[0], input_shape0[1], self.heads, self.filters])
        en_q = torch.permute(en_q, (2, 0, 1, 3))

        en_k = kw
        en_k = torch.reshape(en_k, [input_shape0[0], input_shape0[1], self.heads, self.filters])
        en_k = torch.permute(en_k, (2, 0, 1, 3))

        en_v = vw
        en_v = torch.reshape(en_v, [input_shape0[0], input_shape0[1], self.heads, self.filters])
        en_v = torch.permute(en_v, (2, 0, 1, 3))

        en_a = torch.softmax(torch.einsum('h b i j, h b k j -> h b i k', en_q, en_k) / dk0, dim=-1)
        en_f = torch.einsum('h b i j, h b j k -> h b i k', en_a, en_v)
        en_f = torch.permute(en_f, (1, 2, 0, 3))
        return en_f


class MultiAttentionBlock(nn.Module):

    def __init__(self, filtersins, filters, heads=1):
        super(MultiAttentionBlock, self).__init__()
        self.filtersins = filtersins
        self.filters = filters
        self.heads = heads
        
        self.outchannel_shape = self.filters
        self.encode_dense_q = torch.nn.Linear(self.filtersins[0], self.filters * self.heads, bias=True)
        self.encode_dense_k = torch.nn.Linear(self.filtersins[1], self.filters * self.heads, bias=True)
        self.encode_dense_v = torch.nn.Linear(self.filtersins[2], self.filters * self.heads, bias=True)

        self.attentions = AttentionBlock(self.filters, self.heads)
        self.dense1 = torch.nn.Linear(self.outchannel_shape * self.heads, self.filters, bias=True)
        self.batch0 = torch.nn.InstanceNorm1d(self.outchannel_shape * self.heads, track_running_stats=False, affine=False)
        self.batch1 = torch.nn.InstanceNorm1d(self.outchannel_shape, track_running_stats=False, affine=False)

    def forward(self, inputs):
        qw, kw, vw = inputs
        input_shape0 = qw.shape
        qw = self.encode_dense_q(qw)
        kw = self.encode_dense_k(kw)
        vw = self.encode_dense_v(vw)
        en_f = self.attentions([qw, kw, vw])
        en_f = en_f + torch.reshape(qw, [input_shape0[0], input_shape0[1], self.heads, self.filters])
        en_f = torch.reshape(en_f, [input_shape0[0], input_shape0[1], self.outchannel_shape * self.heads])
        en_f = en_f.permute(0,2,1)
        en_f0 = self.batch0(en_f)
        en_f0 = en_f0.permute(0,2,1)
        en_f = self.dense1(en_f0)
        en_f = en_f + torch.sum(torch.reshape(en_f0, [input_shape0[0], input_shape0[1], self.heads, self.filters]), dim=-2)
        en_f = en_f.permute(0,2,1)
        en_f = self.batch1(en_f)
        en_f = en_f.permute(0,2,1)

        return en_f


class Convde(nn.Module):

    def __init__(self, filtersin, filters, kernel_size=[3, 3], stride=1, padding='same', dilation=(1, 1), groups=1, bias=True):
        super(Convde, self).__init__()
        
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.bias = bias
        #self.fils = filters
        self.filters = filters//2+filters%2
        self.tru_n = filters%2
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        self.dilation = dilation
        self.filtersin = filtersin+3
        self.addc = AddCoords()
        w0 = torch.empty(self.filters, self.filtersin//groups, self.kernel_size[0], self.kernel_size[1])
        nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
        self.w0 = nn.Parameter(w0)
        
        self.kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.filters, self.filters)
        self.new_shape = (self.kernel_size[0] * self.kernel_size[1], self.filtersin, self.filters)
        if self.bias:
            self.bias0 = nn.Parameter(torch.randn(1,self.filters,1,1))
            #self.bias1 = nn.Parameter(torch.randn(1,self.filters,1,1))
        self.dense0 = torch.nn.Linear(self.filters, self.filters, bias=True)
        self.encode_attention = MultiAttentionBlock([self.filters, self.filters, self.filters], self.filters)
        self.decode_attention = MultiAttentionBlock([self.filtersin, self.filters, self.filters], self.filters)
        self.fb_attention = MultiAttentionBlock([self.filtersin, self.filtersin, self.filtersin], self.filters)
        #self.batchc0 = torch.nn.InstanceNorm2d(self.filters, track_running_stats=False, affine=False)

        #self.batchc1 = torch.nn.InstanceNorm2d(filters, track_running_stats=False, affine=False)
        #self.conv_out = nn.Conv2d(filters*2, filters, kernel_size=[1, 1], padding='same', dilation=(1, 1))

    def forward(self, inputs):
        x0 = self.addc(inputs)

        if self.bias:
            output0 = F.conv2d(x0, self.w0, padding=self.padding, dilation=self.dilation) + self.bias0
        else:
            output0 = F.conv2d(x0, self.w0, padding=self.padding, dilation=self.dilation)
        #output0 = F.conv2d(x0, self.w0, padding=self.padding, dilation=self.dilation)
        w = self.w0.permute(2, 3, 1, 0)
        w_v = torch.reshape(w, self.new_shape)
        w_v1 = torch.permute(w_v, (1, 0, 2))
        en_f = self.encode_attention([w_v1, w_v1, w_v1])
        en_f = torch.permute(en_f, (2, 1, 0))
        w_v0 = torch.permute(w_v, (2, 0, 1))
        fb_f = self.fb_attention([w_v0, w_v0, w_v0])
        de_f = self.decode_attention([en_f, fb_f, fb_f])

        de_f = torch.permute(de_f, (1, 2, 0))
        de_f = torch.reshape(de_f, self.kernel_shape)
        de_f = self.dense0(de_f)
        de_f = de_f.permute(3, 2, 0, 1)

        #output0 = self.batchc0(output0)
        #output0 = F.relu(output0)
        output1 = F.conv2d(output0, de_f, stride = self.stride, padding=self.padding, dilation=1, groups=1)
        #output1 = self.conv_out(torch.cat([output0, output1], dim=1))
        if self.tru_n:
            output0 = output0[:,:-1,:,:]
        if self.bias:
            return torch.cat([output0, output1], dim=1)
        return torch.cat([output0, output1], dim=1)