# YOLOv3 common modules

# from _typeshed import Self
import math
import pdb
from copy import copy
from pathlib import Path
from re import X
from matplotlib.pyplot import flag

import numpy as np
import pandas as pd
import requests
from requests import models
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp, device
from torch.nn.parameter import Parameter
from torch.nn import init
from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable
from config import opt

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if True:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def fuseforward(self, x):
#         return self.act(self.conv(x))


class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_outputs):
        input, = ctx.saved_tensors
        grad_inputs = wcgrad(grad_outputs, input)
        # grad_inputs = dorefagrad(grad_outputs, input)
        # grad_inputs = grad_outputs
        return grad_inputs, None


def scale_sign(input):
    return ScaleSigner.apply(input)


class Quantizer(Function):
    @staticmethod
    def forward(ctx, input, nbit):
        ctx.save_for_backward(input)
        scale = float(2 ** nbit - 1)
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_outputs):
        input, = ctx.saved_tensors
        return grad_outputs, None


def quantize(input, nbit):
    return Quantizer.apply(input, nbit)


def wcgrad(grad_inputs, input):
    shape = grad_inputs.shape
    delta = torch.tanh(input)
    delta_col = delta.reshape(-1, 1)
    t = 0.002
    p = torch.matmul(delta_col, torch.matmul(delta_col.t(), grad_inputs.reshape(-1, 1)))
    p = p / torch.matmul(delta_col.t(), delta_col)
    # p = p / torch.max(torch.abs(p))
    grad_outputs = grad_inputs + p.reshape(shape) * t
    return grad_outputs

def ceil_power_of_2(x):
    p = torch.ceil(torch.log(x) / np.log(2.))
    return torch.pow(2.0, p)

def dorefagrad(grad_inputs, input):
    delta = torch.tanh(input)
    grad_outputs = (1 - delta ** 2) * grad_inputs / (torch.max(torch.abs(delta)))
    return grad_outputs


# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
def dorefa_w(x, nbit_w):
    if nbit_w == 1:
        x = scale_sign(x)
    else:
        x = torch.tanh(x)
        x = x / (2 * torch.max(torch.abs(x))) + 0.5
        x = 2 * quantize(x, nbit_w) - 1
    return x

def dorefa_b(input, nbit_b):
    return quantize(torch.clamp(input/16, 0, 1), nbit_b)*16


def dorefa_a(input, nbit_a):
    # pdb.set_trace()
    return quantize(torch.clamp(input/16, 0, 1), nbit_a)*16

def round_power_of_2(x, k=4):
    bound = np.power(2.0, k - 1)
    min_val = np.power(2.0, -bound + 1.0)
    s = torch.sign(x)
    x = torch.clamp(torch.abs(x), min_val, 1.0)
    p = quantize(torch.log(x) / np.log(2.),k)
    return s * torch.pow(2.0, p)

class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_b='dorefa', quan_name_a='dorefa',quan_name_w='dorefa', nbit_w=8 , nbit_b=16,
                 nbit_a=8, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        quan_name_b='dorefa'
        quan_name_w='dorefa'
        quan_name_a='dorefa'
        self.nbit_b = nbit_b
        self.nbit_a = nbit_a
        self.nbit_w = nbit_w
        name_b_dict = {'dorefa': dorefa_b}
        name_a_dict = {'dorefa': dorefa_a}
        name_w_dict = {'dorefa': dorefa_w}
        self.quan_b = name_b_dict[quan_name_b]
        self.quan_a = name_a_dict[quan_name_a]
        self.quan_w = name_w_dict[quan_name_w]
    def forward(self, input):
        if self.nbit_w < 32:
            if opt.shift:
                quant_weight = round_power_of_2(self.weight, self.nbit_w)
            if opt.dorefa:
                quant_weight = self.quan_w(self.weight,self.nbit_w)
            if opt.sign:
                quant_weight = self.quan_w(self.weight,1)
        else:
            quant_weight =self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a)
        else:
            x = input
        output = F.conv2d(x, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)


def reshape_to_bias(input):
    return input.reshape(-1)


class QuantBNFuseConv2d(QuanConv):
    def __init__(
        self,
        in_channels,out_channels,kernel_size,padding,stride=1,dilation=1,bias=False,eps=1e-5,groups= 1,
        momentum = 0.1, nbit_w=32 , nbit_a=32 , nbit_b=16,bn_fuse_calib=True,
        pretrained_model =False,qaft=False,
    ):
        super(QuantBNFuseConv2d,self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            
        )
        self.qaft = qaft
        self.num_flag = 0
        self.pretrained_model = pretrained_model
        self.bn_fuse_calib = bn_fuse_calib
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer(
            'running_mean', torch.zeros((out_channels),dtype=torch.float32)
        )
        self.register_buffer(
            'running_var', torch.ones((out_channels),dtype=torch.float32)
        )
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        quan_name_b='dorefa'
        quan_name_w='dorefa'
        quan_name_a='dorefa'
        self.nbit_b = nbit_b
        self.nbit_a = nbit_a
        self.nbit_w = nbit_w
        name_b_dict = {'dorefa': dorefa_b}
        name_a_dict = {'dorefa': dorefa_a}
        name_w_dict = {'dorefa': dorefa_w}
        self.quan_b = name_b_dict[quan_name_b]
        self.quan_a = name_a_dict[quan_name_a]
        self.quan_w = name_w_dict[quan_name_w]
        self.save = []
    def forward(self,input):
        if not self.qaft:
            # qat, calibrate bn_statis_para
            # 训练态
            # pdb.set_trace()
            if self.training:
                # 先做普通卷积得到A，以取得BN参数
                output = F.conv2d(
                    input,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

                # 更新BN统计参数（batch和running）
                dims = [dim for dim in range(4) if dim != 1]
                batch_mean = torch.mean(output, dim=dims)
                batch_var = torch.var(output, dim=dims)
                with torch.no_grad():
                    if not self.pretrained_model:
                        if self.num_flag == 0:
                            self.num_flag += 1
                            running_mean = batch_mean
                            running_var = batch_var
                        else:
                            # pdb.set_trace()
                            running_mean = (
                                1 - self.momentum
                            ) * self.running_mean + self.momentum * batch_mean
                            running_var = (
                                1 - self.momentum
                            ) * self.running_var + self.momentum * batch_var
                        self.running_mean.copy_(running_mean)
                        self.running_var.copy_(running_var)
                    else:
                        running_mean = (
                            1 - self.momentum
                        ) * self.running_mean + self.momentum * batch_mean
                        running_var = (
                            1 - self.momentum
                        ) * self.running_var + self.momentum * batch_var
                        self.running_mean.copy_(running_mean)
                        self.running_var.copy_(running_var)
                # bn融合
                if self.bias is not None:
                    bias_fused = reshape_to_bias(
                        self.beta
                        + (self.bias - batch_mean)
                        * (self.gamma / torch.sqrt(batch_var + self.eps))
                    )
                else:
                    bias_fused = reshape_to_bias(
                        self.beta
                        - batch_mean * (self.gamma / torch.sqrt(batch_var + self.eps))
                    )  # b融batch
                # bn融合不校准
                if not self.bn_fuse_calib:
                    weight_fused = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(batch_var + self.eps)
                    )  # w融batch
                # bn融合校准
                else:
                    weight_fused = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(self.running_var + self.eps)
                    )  # w融running
            # 测试态
            else:
                if self.bias is not None:
                    bias_fused = reshape_to_bias(
                        self.beta
                        + (self.bias - self.running_mean)
                        * (self.gamma / torch.sqrt(self.running_var + self.eps))
                    )
                else:
                    bias_fused = reshape_to_bias(
                        self.beta
                        - self.running_mean
                        * (self.gamma / torch.sqrt(self.running_var + self.eps))
                    )  # b融running
                weight_fused = self.weight * reshape_to_weight(
                    self.gamma / torch.sqrt(self.running_var + self.eps)
                )  # w融running
        else:
            # qaft, freeze bn_statis_para
            if self.bias is not None:
                bias_fused = reshape_to_bias(
                    self.beta
                    + (self.bias - self.running_mean)
                    * (self.gamma / torch.sqrt(self.running_var + self.eps))
                )
            else:
                bias_fused = reshape_to_bias(
                    self.beta
                    - self.running_mean
                    * (self.gamma / torch.sqrt(self.running_var + self.eps))
                )  # b融running
            weight_fused = self.weight * reshape_to_weight(
                self.gamma / torch.sqrt(self.running_var + self.eps)
            )  # w融running
        if self.nbit_w < 32:
            if opt.shift:
                quant_weight = round_power_of_2(weight_fused, self.nbit_w)
            if opt.dorefa:
                quant_weight = self.quan_w(weight_fused,self.nbit_w)
            if opt.sign:
                quant_weight =self.quan_w(weight_fused,1)
        else:
            quant_weight =weight_fused
        
        if self.nbit_a < 32:
            quant_input = self.quan_a(input,self.nbit_a)
        else:
            quant_input = input
        if self.nbit_b < 32:
            quan_bias = self.quan_b(bias_fused,self.nbit_b)
        if not self.qaft:
            # qat, quant_bn_fuse_conv
            # 量化卷积
            if self.training:  # 训练态
                # bn融合不校准
                if not self.bn_fuse_calib:
                    output = F.conv2d(
                        quant_input,
                        quant_weight,
                        quan_bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                # bn融合校准
                else:
                    output = F.conv2d(
                        quant_input,
                        quant_weight,
                        None,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )  # 注意，这里不加bias（self.bias为None）
                    # （这里将训练态下，卷积中w融合running参数的效果转为融合batch参数的效果）running ——> batch
                    output *= reshape_to_activation(
                        torch.sqrt(self.running_var + self.eps)
                        / torch.sqrt(batch_var + self.eps)
                    )
                    output += reshape_to_activation(quan_bias)
            else:  # 测试态
                output = F.conv2d(
                    quant_input,
                    quant_weight,
                    quan_bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )  # 注意，这里加bias，做完整的conv+bn
        else:
            # qaft, quant_bn_fuse_conv
            output = F.conv2d(
                quant_input,
                quant_weight,
                quan_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,stride=1,padding=None,groups=1,act = True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv,self).__init__()
        # pdb.set_trace()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size, stride , autopad(kernel_size, padding), groups=groups, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels,eps=1e-5, momentum=0.1)
        self.act = nn.LeakyReLU(1/64) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())      
    def forward(self, x):        
        out1 = self.conv(x)
        # out2 =self.bn(out1)
        out = self.act(out1)
        return out


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, groups=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        
        self.cv2 = Conv(c_, c2, 3, 1, groups=groups)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # pdb.set_trace()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
        # pdb.set_trace()

    def forward(self, x):
        # pdb.set_trace()

        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)


class AutoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super(AutoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.csom/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv3 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = Conv(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

def prepare(
    model,
    inplace=False,
    nbit_a=8,
    nbit_w=8,
    bn_fuse = False,
    bn_fuse_calib=False,
    pretrained_model=False,
    qaft=False,
):
    if not inplace:
        model = copy.deepcopy(model)
    add_quant_op(
        model,
        nbit_a=opt.nbit_a,
        nbit_w=opt.nbit_w,
        bn_fuse = bn_fuse,
        bn_fuse_calib=bn_fuse_calib,
        pretrained_model=pretrained_model,
        qaft=qaft,
    )
    return model


def add_quant_op(
    module,
    nbit_a=8,
    nbit_w=8,
    bn_fuse = False,
    bn_fuse_calib=False,
    pretrained_model=False,
    qaft=False,
): 
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if bn_fuse:
                conv_name_temp = name
                conv_child_temp = child
            else:
                conv_name_temp = name
                conv_child_temp = child
                if child.bias is not None:
                    quant_conv = QuanConv(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        nbit_w = opt.nbit_w,
                        nbit_a = opt.nbit_a
                    )
                else:
                    quant_conv = QuanConv(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        nbit_w = opt.nbit_w,
                        nbit_a = opt.nbit_a
                    )
                    module._modules[conv_name_temp] = quant_conv
        elif isinstance(child, nn.BatchNorm2d):
            pdb.set_trace()
            if bn_fuse:
                if conv_child_temp.in_channels !=3:                        
                    if conv_child_temp.bias is not None:
                        quant_bn_fuse_conv = QuantBNFuseConv2d(
                            conv_child_temp.in_channels,
                            conv_child_temp.out_channels,
                            conv_child_temp.kernel_size,
                            stride=conv_child_temp.stride,
                            padding=conv_child_temp.padding,
                            dilation=conv_child_temp.dilation,
                            groups=conv_child_temp.groups,
                            bias=True,
                            eps=child.eps,
                            momentum=child.momentum,
                            nbit_a=nbit_a,
                            nbit_w=nbit_w,
                            pretrained_model=pretrained_model,
                            qaft=qaft,
                            bn_fuse_calib=bn_fuse_calib,
                        )
                        quant_bn_fuse_conv.bias.data = conv_child_temp.bias
                    else:
                        quant_bn_fuse_conv = QuantBNFuseConv2d(
                            conv_child_temp.in_channels,
                            conv_child_temp.out_channels,
                            conv_child_temp.kernel_size,
                            stride=conv_child_temp.stride,
                            padding=conv_child_temp.padding,
                            dilation=conv_child_temp.dilation,
                            groups=conv_child_temp.groups,
                            bias=False,
                            eps=child.eps,
                            momentum=child.momentum,
                            nbit_a=nbit_a,
                            nbit_w=nbit_w,
                            pretrained_model=pretrained_model,
                            qaft=qaft,
                            bn_fuse_calib=bn_fuse_calib,
                        )               
                    quant_bn_fuse_conv.weight.data = conv_child_temp.weight
                    quant_bn_fuse_conv.gamma.data = child.weight
                    quant_bn_fuse_conv.beta.data = child.bias
                    quant_bn_fuse_conv.running_mean.copy_(child.running_mean)
                    quant_bn_fuse_conv.running_var.copy_(child.running_var)
                    # pdb.set_trace()
                    # quant_bn_fuse_conv.running_mean.data = child.running_mean
                    # quant_bn_fuse_conv.running_var.data = child.running_var
                    module._modules[conv_name_temp] = quant_bn_fuse_conv
                    module._modules[name] = nn.Identity()
                else:                     
                        if conv_child_temp.bias is not None:
                            quant_bn_fuse_conv = QuantBNFuseConv2d(
                                conv_child_temp.in_channels,
                                conv_child_temp.out_channels,
                                conv_child_temp.kernel_size,
                                stride=conv_child_temp.stride,
                                padding=conv_child_temp.padding,
                                dilation=conv_child_temp.dilation,
                                groups=conv_child_temp.groups,
                                bias=True,
                                eps=child.eps,
                                momentum=child.momentum,
                                nbit_a=32,
                                nbit_w=nbit_w,
                                pretrained_model=pretrained_model,
                                qaft=qaft,
                                bn_fuse_calib=bn_fuse_calib,
                            )
                            quant_bn_fuse_conv.bias.data = conv_child_temp.bias
                        else:
                            quant_bn_fuse_conv = QuantBNFuseConv2d(
                                conv_child_temp.in_channels,
                                conv_child_temp.out_channels,
                                conv_child_temp.kernel_size,
                                stride=conv_child_temp.stride,
                                padding=conv_child_temp.padding,
                                dilation=conv_child_temp.dilation,
                                groups=conv_child_temp.groups,
                                bias=False,
                                eps=child.eps,
                                momentum=child.momentum,
                                nbit_a=32,
                                nbit_w=nbit_w,
                                pretrained_model=pretrained_model,
                                qaft=qaft,
                                bn_fuse_calib=bn_fuse_calib,
                            )               
                        quant_bn_fuse_conv.weight.data = conv_child_temp.weight
                        quant_bn_fuse_conv.gamma.data = child.weight
                        quant_bn_fuse_conv.beta.data = child.bias
                        # quant_bn_fuse_conv.running_mean.copy_(child.running_mean)
                        # quant_bn_fuse_conv.running_var.copy_(child.running_var)
                        quant_bn_fuse_conv.running_mean.data = child.running_mean
                        quant_bn_fuse_conv.running_var.data = child.running_var
                        module._modules[conv_name_temp] = quant_bn_fuse_conv
                        module._modules[name] = nn.Identity()
        else:
            add_quant_op(
                child,
                nbit_a=nbit_a,
                nbit_w=nbit_w,
                bn_fuse=bn_fuse,
                bn_fuse_calib=bn_fuse_calib,
                pretrained_model=pretrained_model,
            )
