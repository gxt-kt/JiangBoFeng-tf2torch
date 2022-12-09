import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from darknet19.StoreWeights_BinConvert import *
def RoundPower2_pytorch(x, k=4):#浮点数转换为指数

  bound = np.power(2.0, k - 1)
  min_val = np.power(2.0, -bound + 1.0)
  s = torch.sign(x)
  # # Check README.md for why `x` need to be divided by `8`
  # x = np.clip(np.absolute(x / 8), min_val, 1.0)
  # x = torch.clamp(torch.abs(x), min_val, 1.0)
  x = torch.clamp(torch.abs(x*64), min_val, 1.0)
  x = torch.clamp(torch.abs(x/64), min_val, 1.0)
  # Temporary. `*8` during inference and don't change during convert.
  # In fact, it should be `/8` during convert and don't change during inference.
  # p = torch.round(torch.log(x) / torch.log(2.0 * torch.ones(x.size())))
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
    def  __init__(self,in_channels, out_channels, kernel_size,
                 #bias,
                 stride,padding=0, #dilation, groups,
                 quan_name='shift',
                 w_integer=4,w_bit=4,a_integer=3,a_bit=8,
                 b_integer=7,b_bit=16):
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
                weight_quant_ex,weight_sign,weight_for_save=RoundPower2_pytorch(self.weight,self.w_integer)
                weight_quant=weight_sign*torch.pow(2,weight_quant_ex)
                bias_quant=Round2Fixed_tensor(self.bias,self.b_integer,self.b_bit)

            else:
                weight_quant = self.weight
                bias_quant = self.bias
                print('Qutification in other ways.')
        else:
            weight_quant = self.weight
            bias_quant=self.bias
            print("NO quantification method")

        if self.a_bit <32:
            activation = Round2Fixed_tensor(input, self.a_integer, self.a_bit)
        else:
            activation=input

        output=F.conv2d(activation,weight=weight_quant,bias=bias_quant,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        output_lk=F.leaky_relu(output,0.125)
        with open(str(weight_path), mode='ab') as f:
            print('Store weight ')
            Store4DBinConvert(weight_for_save, f)
        return output

class YoloTinyModel(nn.Module):
    def __init__(self):
        super(YoloTinyModel, self).__init__()
        self.conv1=QuanConv(in_channels=8,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv3=QuanConv(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv5=QuanConv(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv7=QuanConv(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.conv9=QuanConv(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.conv11=QuanConv(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)#why this need padding??
        self.conv13=QuanConv(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.conv14=QuanConv(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0)
        self.conv15=QuanConv(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv16=QuanConv(in_channels=512,out_channels=256,kernel_size=1,stride=1,padding=0)

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

        return x1l,x16




def Img2np(img):
  img_o = np.reshape(img, (1, img_channels, img_size, img_size))
  return img_o

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


  data_img_quantize=Round2Int(data_img,3,8)
  return data_img_quantize.astype(np.int32)


def ConvertTensor(tensor, axis, paral):
  '''
  Brief:
    Return a flattened 1D tensorflow tensor.

    For weight tensor, axis should be set to [3, 2, 0, 1, 4], which is
      (row, col, in_channels, out_channels/paral, paral) ->
      (out_channels/paral, in_channels, row, col, paral)

    For image tensor, axis should be set to [2, 0, 3, 1, 4]
      (row, col, 1, ch/paral, paral) -> (1, row, ch/paral, col, paral)
  '''
  if len(tensor.shape) == 4:
    # print('[INFO][utils.py] Convert 4D tensor {}'.format(tensor.name))
    print('[INFO][utils.py] Convert 4D tensor.')
    height = tensor.shape[0]
    width = tensor.shape[1]
    dim = tensor.shape[2]
    tensor_paral = torch.reshape(
        tensor, [height, width, dim, -1, paral])
    # tensor_transpose = tfTranspose(tensor_paral, [3, 2, 0, 1, 4])
    tensor_transpose = tensor_paral.permute(axis)
    tensor_1d = torch.reshape(tensor_transpose, [-1])
  elif len(tensor.shape) == 1:
    # print('[INFO][utils.py][ConvertTensor] 1D tensor {}, '
        # 'no need to be converted'.format(tensor.name))
    print('[INFO][utils.py][ConvertTensor] 1D tensor, no need to be converted')
    tensor_1d = tensor
  else:
    print('[INFO][utils.py][ConvertTensor] Wrong tensor shape!!!')
    tensor_1d = None

  return tensor_1d
def SaveFeatureMap(data_path, FeatureMap):# save as float
    FeatureMap_tranpose=FeatureMap.permute(2,3,0,1)
    FeatureMap_1d=ConvertTensor(FeatureMap_tranpose,[2,4,3,0,1],1)
    FeatureMap_fix=Round2Fixed_np(FeatureMap_1d.detach().numpy(),7,16)
    channels=FeatureMap.shape[1]
    height=FeatureMap.shape[2]
    width=FeatureMap.shape[3]

    img_reshape=np.reshape(FeatureMap_fix,(1,channels,height,width))
    img_transpose=np.transpose(img_reshape, [2, 3, 0, 1])
    data_1d = ConvertTensor(torch.from_numpy(img_transpose), [2, 0, 3, 1, 4], 8)

    # data_quantize = Round2Int(data_1d.numpy(), 3, 8).astype(np.int32)
    # data_quantize=Round2Int(data_1d.numpy(),3,8).astype(np.float32)
    data_quantize=FeatureMap_fix
    with open(str(data_path), mode='wb') as f:
        print('Store' + str(data_path))
        for npiter in np.nditer(data_quantize):
          f.write(npiter)
        f.close()
def Round2Fixed_np(x, integer=16, k=32):#浮点数转为定点数,
  #Use this method when compute for meddle answer
  assert integer >= 1, integer
  fraction = k - integer
  bound = np.power(2.0, integer - 1)#this is real bound
  n = np.power(2.0, fraction)
  min_val = -bound
  # max_val = bound
  max_val=bound- 1./np.power(2.,fraction)
  # ??? Is this round function correct
  x_round = np.floor(x * n)/n
  clipped_value = np.clip(x_round, min_val, max_val)
  return clipped_value

def Round2Int(x, integer=16, k=32):
  # Convert at to int32,and save as int 32 which is n times as original.
  # Only used when need output as FPGA
  assert integer >= 1, integer
  fraction = k - integer
  bound = np.power(2.0, integer - 1)#this is int bound, not real bound
  n = np.power(2.0, fraction)
  min_val = -bound
  max_val = bound-1./np.power(2.,fraction)
  # correct the fuction
  x_round = np.floor(x * n)/n
  clipped_value = np.clip(x_round, min_val, max_val)
  return clipped_value*n

def Save_fix(out,output_path):
    out_other=Convert_Map(out.detach().numpy())
    with open(str(output_path), mode='wb') as f:#save as fix
        print('Store' + str(output_path))
        for npiter in np.nditer(out_other):
          f.write(npiter)
        f.close()

if __name__ == '__main__':
    model=YoloTinyModel()
    model.load_state_dict(torch.load('yolo_tiny_weights_RIGHT.pth'))
    out1_path=Path('.')/'out'/'out_64_208_208_small.bin'
    out15_path = Path('.') / 'out'/'out_256_13_13_small_float.bin'
    weight_path='check_weight/yolo_bench.bin'#check if the weight is right
    img_channels = 8
    img_size =416
    image_bin_path = 'img_416_8_small.bin'
    img = np.fromfile(image_bin_path, dtype=np.float32)
    img_np = Img2np(img)
    img_tensor = torch.from_numpy(img_np)
    # img_quantize = torch.from_numpy(Round2Fixed_np(img_np, 3, 8))
    out1,out15=model(img_tensor)
    data_path=Path('.') / 'out'/'out_256_13_13_small_fix.bin'


    Save_fix(out15,data_path)

    # out15_other=Convert_Map(out15.detach().numpy())
    # with open(str(data_path), mode='wb') as f:#save as fix
    #     print('Store' + str(data_path))
    #     for npiter in np.nditer(out15_other):
    #       f.write(npiter)
    #     f.close()
    # data_path_1 = Path('.') / 'out_208_208_64_fix.bin'
    # with open(str(data_path_1), mode='wb') as f:
    #     print('Store' + str(data_path))
    #     for npiter in np.nditer(out15_other):
    #         f.write(npiter)
    #     f.close()


    SaveFeatureMap(out15_path,out15)
    # SaveFeatureMap(out1_path,out1)
