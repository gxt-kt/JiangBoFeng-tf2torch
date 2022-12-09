import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from yolo_input import Round2Int,Save_fix,SaveFeatureMap
from pathlib import Path
import argparse
from StoreWeights_BinConvert import Store4DBinConvert,Store1DBinConvert
def RoundPower2_pytorch_dark(input, k=4):#浮点数转换为指数
    x=input
    bound = np.power(2.0, k - 1)
    min_val = np.power(2.0, -bound + 1.0)
    s = torch.sign(x)
    # # Check README.md for why `x` need to be divided by `8`
    # x = np.clip(np.absolute(x / 8), min_val, 1.0)
    x = torch.clamp(torch.abs(x), min_val, 1.0)
    # x = torch.clamp(torch.abs(x*64), min_val, 1.0)
    # x = torch.clamp(torch.abs(x/64), min_val, 1.0)
    # Temporary. `*8` during inference and don't change during convert.
    # In fact, it should be `/8` during convert and don't change during inference.
    p = torch.round(torch.log(x) / np.log(2.))
    # p = torch.round(torch.log(x*8) / torch.log(2.0*torch.ones(x.size())))

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
  clipped_value = torch.clamp(x_round, min_val, max_val)
  return clipped_value

def Roundtofix(x,integer=16,nbit=32):
    assert integer>= 1
    fraction = nbit - integer
    bound = np.power(2,integer-1)
    n = np.power(2,fraction)
    min_val = -bound
    max_val = bound - 1./n
    x_round = torch.floor(x*n)/n
    clipped_value = torch.clamp(x_round,min_val,max_val)
    return clipped_value

class QuanConv(nn.Conv2d):
    def  __init__(self,in_channels, out_channels, kernel_size,
                 #bias,
                 stride,padding=0, #dilation, groups,
                 quan_name='shift',
                 w_integer=4,w_bit=4,a_integer=3,a_bit=8,
                 b_integer=4,b_bit=16):#change !!!
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
        self.b_integer=b_integer
        self.b_bit=b_bit
    def forward(self,input):
        if self.w_bit<32:
            if self.quan_name=='shift':
                weight_quant_ex,weight_sign,weight_forsave=RoundPower2_pytorch_dark(self.weight,self.w_integer)
                weight_quant=weight_sign*torch.pow(2,weight_quant_ex)
                bias_quant=Roundtofix(self.bias,self.b_integer,self.b_bit)
                n=np.power(2,self.b_bit-self.b_integer)
                bias_forsave=bias_quant*n
                with open(str(bias_path), mode='ab') as f:
                    print('Store bias ')
                    Store1DBinConvert(bias_forsave, f)
                with open(str(weight_path), mode='ab') as f:
                    print('Store weight ')
                    Store4DBinConvert(weight_forsave, f)

            else:
                weight_quant = self.weight
                bias_quant = self.bias
                weight_forsave=weight_quant
                bias_forsave=bias_quant
                print('Qutification in other ways.')
        else:
            weight_quant = self.weight
            bias_quant=self.bias
            weight_forsave = weight_quant
            bias_forsave = bias_quant
            print("NO quantification method")

        if self.a_bit <32:
            activation = Roundtofix(input, self.a_integer, self.a_bit)
        else:
            activation=input

        output=F.conv2d(activation,weight=weight_quant,bias=bias_quant,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        output_lk=F.leaky_relu(output,0.125)
        # self.weight=weight_forsave
        # self.bias=bias_forsave
        return output_lk

class Darknet_19_Model(nn.Module):
    def __init__(self):
        super(Darknet_19_Model, self).__init__()
        self.conv0=QuanConv(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1=QuanConv(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv2=QuanConv(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv3=QuanConv(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv4=QuanConv(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv5=QuanConv(in_channels=128,out_channels=64,kernel_size=1,stride=1)
        self.conv6=QuanConv(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv7=QuanConv(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.conv8=QuanConv(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv9=QuanConv(in_channels=256,out_channels=128,kernel_size=1,stride=1)
        self.conv10=QuanConv(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv11=QuanConv(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.conv12=QuanConv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13=QuanConv(in_channels=512,out_channels=256,kernel_size=1,stride=1)
        self.conv14=QuanConv(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv15=QuanConv(in_channels=512,out_channels=256,kernel_size=1,stride=1)
        self.conv16=QuanConv(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv17 = QuanConv(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv18 = QuanConv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv19 = QuanConv(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.conv20 = QuanConv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv21 = QuanConv(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        self.conv22 = QuanConv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv23 = QuanConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv24 = QuanConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv25 = QuanConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        # self.detect = QuanConv(in_channels=1024, out_channels=255, kernel_size=1, stride=1)
        self.detect = QuanConv(in_channels=1024, out_channels=256, kernel_size=1, stride=1)

    def forward(self,input):
        x0=self.conv0(input)
        x1=self.conv1(x0)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)
        x6=self.conv6(x5)
        x7=self.conv7(x6)
        x8=self.conv8(x7)
        x9=self.conv9(x8)
        x10=self.conv10(x9)
        x11=self.conv11(x10)
        x12=self.conv12(x11)
        x13=self.conv13(x12)
        x14=self.conv14(x13)
        x15=self.conv15(x14)
        x16=self.conv16(x15)
        x17=self.conv17(x16)
        x18=self.conv18(x17)
        x19=self.conv19(x18)
        x20=self.conv20(x19)
        x21=self.conv21(x20)
        x22=self.conv22(x21)
        x23=self.conv23(x22)
        x24=self.conv24(x23)
        x25=self.conv25(x24)
        x_detect=self.detect(x25)

        return x0,x1,x2,x3,x25,x_detect

def Convert_Map(re_map):
  img_w = re_map.shape[-1]
  img_h = re_map.shape[-2]
  img_ch = re_map.shape[-3]
  PARAL_IN = 8
  re_map=re_map.reshape(img_ch,img_h,img_w)

  num_pixel = img_w * img_h * img_ch
  sh = img_ch * img_w
  sc = img_w * PARAL_IN
  sw = PARAL_IN
  sp = 1

  data_img = np.zeros(num_pixel, dtype=np.float32)

  for row in range(img_h):
    for k in range(int(img_ch / PARAL_IN)):
      for col in range(img_w):
        for p in range(PARAL_IN):
          data_img[row * sh + k * sc + col * sw + p * sp] = \
            re_map[k * PARAL_IN + p, row, col]


  return data_img

def CreatImg(chs=8,size=416):
    img_3=torch.rand(3,size,size)-0.5
    img_0=torch.zeros(chs-3,size,size)
    newimg=torch.cat([img_3,img_0],dim=0)
    return newimg


def SaveTensor2Fix(out,int,k,output_path):
    out_1d=Convert_Map(out.detach().numpy())
    out_other=Round2Int(out_1d,int,k).astype(np.int32)
    with open(str(output_path), mode='wb') as f:#save as fix
        print('Store ' + str(output_path))
        for npiter in np.nditer(out_other):
          f.write(npiter)
        f.close()


def merage_bin(bin_file_merage,bin_file_1,bin_file_2):
    file_merage = open(bin_file_merage, 'wb')
    file_1 = open(bin_file_1, 'rb')
    data =file_1 .read()
    file_merage.write(data)
    file_2 = open(bin_file_2, 'rb')
    data = file_2.read()
    file_merage.write(data)
    file_1.close()
    file_2.close()
    file_merage.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CNN inference for fpga.'
    'Including merge bn layer. Store output at the end')

    # img
    parser.add_argument(
        '--input_path', default='data/images/416_img_8chs_new.pth',
        help='Image file name.')
    parser.add_argument(
        '--input_path_fix', default='input_darknet_fix_416.bin',
        help='Save image as fix.')
    parser.add_argument(
        '--input_path_float', default='input_darknet_float_416.bin',
        help='Save image as float.')
    parser.add_argument(
        '--img_size', default=416, type=int,
        help='Save image size.')
    parser.add_argument(
        '--img_channels', default=8, type=int,
        help='Image channels.')

    parser.add_argument(
        '--weights_bias', default='darknet_19_weights_order_new.pt',
        help='weights and bias for checkpoints . Default is .pth')
    parser.add_argument(
        '--weight_store_path', default='darknet_19_weight.bin',
        help='weights and bias for save . darknet_19_weight.bin')
    parser.add_argument(
        '--bias_store_path', default='darknet_19_bias.bin',
        help='weights and bias for save . darknet_19_bias.bin')
    parser.add_argument(
        '--final_weights_store_path', default='final_darknet_19_weights.bin',
        help='weights and bias for save . darknet_19_bias.bin')

    parser.add_argument(
        '--out_path_x0_fix', default='output/output_darknet_x0_fix_416.bin',
        help='Output file name. Default is output_darknet_fix.bin')
    parser.add_argument(
        '--out_path_x0_float', default='output/output_darknet_x0_float_416.bin',
        help='Output file name. Default is output_darknet_float.bin')

    parser.add_argument(
        '--out_path_x25_fix', default='output/output_darknet_x25_fix_416.bin',
        help='Output file name. Default is output_darknet_fix.bin')
    parser.add_argument(
        '--out_path_x25_float', default='output/output_darknet_x25_float_416.bin',
        help='Output file name. Default is output_darknet_float.bin')

    parser.add_argument(
        '--out_path_detect_fix', default='output/output_darknet_detect_fix_416.bin',
        help='Output file name. Default is output_darknet_fix.bin')
    parser.add_argument(
        '--out_path_detect_float', default='output/output_darknet_detect_float_416.bin',
        help='Output file name. Default is output_darknet_float.bin')
    args = parser.parse_args()



    input_path=args.input_path
    input_path_fix=Path('.')/'fig_out'/args.input_path_fix
    input_path_float=Path('.')/'fig_out'/args.input_path_float

    out_path_x0_fix=args.out_path_x0_fix
    out_path_x0_float=args.out_path_x0_float

    out_path_x25_fix=args.out_path_x25_fix
    out_path_x25_float=args.out_path_x25_float
    out_path_detect_fix=args.out_path_detect_fix
    out_path_detect_float=args.out_path_detect_float
    #Create a Model
    model=Darknet_19_Model()

    #img input and save
    img_channels = args.img_channels
    img_size =args.img_size
    # img=CreatImg(img_channels,img_size)
    # torch.save(img,'img_tensor')
    img=torch.load(input_path)
    img_fix=Roundtofix(img,7,16)
    SaveTensor2Fix(img,3,8,input_path_fix)#3,8
    SaveFeatureMap(input_path_float,img)

    #load inference model
    model_path = Path('.')/'model_weights'/args.weights_bias
    sta=torch.load(model_path)
    model.load_state_dict(sta)
    weight_path= Path('.')/'model_weights'/args.weight_store_path
    with open(str(weight_path), mode='wb') as f:
        f.truncate()
    bias_path= Path('.')/'model_weights'/args.bias_store_path
    with open(str(bias_path), mode='wb') as f:
        f.truncate()
    final_weights_path= Path('.')/'model_weights'/args.final_weights_store_path
    with open(str(final_weights_path), mode='wb') as f:
        f.truncate()

    out0,out1,out2,out3,out25,out_detect=model(img_fix)
    # StoreWeightBinConvert(model.state_dict(),weights_path)
    # torch.save(out0,'darknet_x1.pt')
    # torch.save(out1,'darknet_x2.pt')
    # torch.save(out2,'darknet_x3.pt')
    # torch.save(out3,'darknet_x4.pt')
    torch.save(out25, 'darknet_x25.pt')
    Save_fix(out0,out_path_x0_fix)#3,8
    SaveFeatureMap(out_path_x0_float,out0)

    Save_fix(out25,out_path_x25_fix)#3,8
    SaveFeatureMap(out_path_x25_float,out25)

    Save_fix(out_detect,out_path_detect_fix)#3,8
    SaveFeatureMap(out_path_detect_float,out_detect)

    merage_bin(final_weights_path,bias_path,weight_path)