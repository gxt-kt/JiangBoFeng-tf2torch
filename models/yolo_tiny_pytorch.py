import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
def RoundPower2_pytorch(x, k=4):#浮点数转换为指数

  bound = np.power(2.0, k - 1)
  min_val = np.power(2.0, -bound + 1.0)
  s = torch.sign(x)
  # # Check README.md for why `x` need to be divided by `8`
  # x = np.clip(np.absolute(x / 8), min_val, 1.0)
  # x = torch.clamp(torch.abs(x), min_val, 1.0)
  x = torch.clamp(torch.abs(x*64), min_val, 1.0)
  x = torch.clamp(torch.abs(x/64), min_val, 1.0)
  p = torch.round(torch.log(x*8) / torch.log(2.0*torch.ones(x.size())))
  # sign_judge=t.eq(s,-t.ones(x.size()))
  # p = 0x8*sign_judge + p.type(dtype=t.CharTensor)
  p_1=(-4*(s-1)-p).type(dtype=torch.CharTensor)
  return p,s,p_1

def Round2Fixed_tensor(x, integer=16, k=32):#浮点数转为定点数,
  #Use this method when compute for meddle answer
  assert integer >= 1, integer
  x_shape=x.shape
  base=2.0*torch.ones(x_shape)
  fraction = k - integer
  bound = np.power(base, integer - 1)#this is real bound
  n = torch.pow(base, fraction)
  min_val = -bound
  # max_val = bound
  max_val=bound- 1./n
  # ??? Is this round function correct
  x_round = torch.floor(x * n)/n
  clipped_value = torch.clip(x_round, min_val, max_val)
  return clipped_value

class QuanConv(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size,
                 #bias,
                 stride,padding, #dilation, groups,
                 quan_name='shift',w_integer=4,w_bit=4,a_integer=3,a_bit=8):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding#dilation, groups,
            # quan_name, w_integer, w_bit, a_integer, a_bit
            #bias
        )
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.quan_name=quan_name
        self.stride=stride
        self.padding=padding
        self.w_integer=w_integer
        self.w_bit=w_bit
        self.a_integer=a_integer
        self.a_bit=a_bit
    def forward(self,input):
        if self.w_bit<32:
            if self.quan_name=='shift':
                weight_quant_ex,weight_sign,weight_for_save=RoundPower2_pytorch(self.weight,self.w_integer)
                weight_quant=weight_sign*torch.pow(2,weight_quant_ex)
                bias_quant=Round2Fixed_tensor(self.bias,self.a_integer,self.a_bit)
            else:
                weight_quant = self.weight
                bias_quant = self.bias
                print('Qutification in other ways.')
        else:
            weight_quant = self.weight
            bias_quant=self.bias
            print("NO quantification method")

        if self.a_bit <32:
            x=Round2Fixed_tensor(input,self.a_integer,self.a_bit)
        else:
            x=input

        output=F.conv2d(x,weight=weight_quant,bias=bias_quant,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        #output_lk=F.leaky_relu(output,1./64)
        return output

class YoloTinyModel(nn.Module):
    def __init__(self):
        super(YoloTinyModel, self).__init__()
        self.conv1=QuanConv(in_channels=8,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv3=QuanConv(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv5=QuanConv(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.conv7=QuanConv(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.conv9=QuanConv(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1)
        self.conv11=QuanConv(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1)#why this need padding??
        self.conv13=QuanConv(in_channels=1024,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv14=QuanConv(in_channels=256,out_channels=512,kernel_size=1,stride=1,padding=0)
        self.conv15=QuanConv(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv16=QuanConv(in_channels=256,out_channels=512,kernel_size=1,stride=1,padding=0)

    def forward(self,input):
        x1=self.conv1(input)
        x1l=F.leaky_relu(x1,1./64)
        x3=self.conv3(x1l)
        x3l=F.leaky_relu(x3,1./64)
        x5=self.conv5(x3l)
        x5l=F.leaky_relu(x5,1./64)
        x7=self.conv7(x5l)
        x7l=F.leaky_relu(x7,1./64)
        x9=self.conv9(x7l)
        x9l=F.leaky_relu(x9,1./64)
        x11=self.conv11(x9l)
        x11l=F.leaky_relu(x11,1./64)
        x13=self.conv13(x11l)
        x13l=F.leaky_relu(x13,1./64)
        x14=self.conv14(x13l)
        x14l=F.leaky_relu(x14,1./64)
        x15=self.conv15(x14l)
        x15l=F.leaky_relu(x15,1./64)
        x16=self.conv16(x15l)

        return x1l,x15l


