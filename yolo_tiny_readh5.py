import torch
from pathlib import Path
from models.yolo_tiny import GenerateModel
import tensorflow as tf
import numpy as np

def readh5_pth(h5_weights):
  weight_tensortf = tf.convert_to_tensor(h5_weights)
  weight_npraw = weight_tensortf.numpy()
  if weight_npraw.shape==4:
      weight_np = np.transpose(weight_npraw, (3, 2, 0, 1))  # tf weights(h,w,in,out),torch weights(out,in,h,w)
  else:
      weight_np = weight_npraw
  weight_tensor = torch.from_numpy(weight_np)
  return weight_tensor

if __name__ == '__main__':
    weight_path=Path('.')/'yolo_tiny_model'/'yolo_tiny_bench1.h5'
    weight_out_path=Path('.')/'yolo_tiny_model'/'yolo_tiny_weights_RIGHT.pth'
    quantize='shift'
    quantize_w_int=4
    quantize_w=4
    quantize_x_int=3
    quantize_x=8

    img_h=416
    img_w=416
    img_ch=8

    #model = GenerateModel(quantize, quantize_w_int, quantize_w, quantize_x_int, quantize_x)
    model = GenerateModel('shift', 4, 4, 3, 8)
    input_tensor_shape = (None, img_h,img_w,img_ch)
    model.build(input_tensor_shape)
    model.load_weights(str(weight_path))
    sta=model.weights


    w1=readh5_pth(sta[0]).permute(3,2,0,1)
    b1=readh5_pth(sta[1])
    w2=readh5_pth(sta[3]).permute(3,2,0,1)
    b2=readh5_pth(sta[4])
    w3=readh5_pth(sta[5]).permute(3,2,0,1)
    b3=readh5_pth(sta[6])
    w4=readh5_pth(sta[7]).permute(3,2,0,1)
    b4=readh5_pth(sta[8])
    w5=readh5_pth(sta[9]).permute(3,2,0,1)
    b5=readh5_pth(sta[10])
    w6_bench1=readh5_pth(sta[11]).permute(3,2,0,1)
    b6_bench1=readh5_pth(sta[12])
    w7_bench1=readh5_pth(sta[13]).permute(3,2,0,1)
    b7_bench1=readh5_pth(sta[14])
    w8_bench1=readh5_pth(sta[15]).permute(3,2,0,1)
    b8_bench1=readh5_pth(sta[16])
    w9_bench1=readh5_pth(sta[17]).permute(3,2,0,1)
    b9_bench1=readh5_pth(sta[18])
    w10_bench1=readh5_pth(sta[19]).permute(3,2,0,1)
    b10_bench1=readh5_pth(sta[20])

    yolo_tiny_model_weights={'conv1.weight':w1,'conv3.weight':w2,'conv5.weight':w3,'conv7.weight':w4,'conv9.weight':w5,'conv11.weight':w6_bench1,'conv13.weight':w7_bench1,'conv14.weight':w8_bench1,'conv15.weight':w9_bench1,'conv16.weight':w10_bench1,
                             'conv1.bias':b1,   'conv3.bias':b2, 'conv5.bias':b3,  'conv7.bias':  b4, 'conv9.bias':b5,  'conv11.bias':b6_bench1,   'conv13.bias':b7_bench1,  'conv14.bias':b8_bench1, 'conv15.bias':b9_bench1,  'conv16.bias':b10_bench1}
    torch.save(yolo_tiny_model_weights,weight_out_path)

