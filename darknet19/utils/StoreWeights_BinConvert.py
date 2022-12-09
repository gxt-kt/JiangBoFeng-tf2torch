import numpy as np
def changelist(list_o):
  length=len(list_o)
  list_n=[None]*length
  halfl=int(length/2)
  for i in range(halfl):
    list_n[2*i+1]=list_o[2*i]
    list_n[2*i]=list_o[2*i+1]
  return list_n
def StoreWeightBinConvert(weight_dict, outpath):
  len_dict=len(weight_dict)
  print(len_dict)
  namelist=changelist(list(weight_dict.keys()))
  with open(str(outpath), mode='wb') as f:
    for i in range(len_dict):
      layer_name=namelist[i]
      weight=weight_dict[layer_name]
      if len(weight.shape) == 4:
        print('[INFO][utils.py] Store 4D tensor {}'.format(layer_name))
        Store4DBinConvert(weight, f)
      elif len(weight.shape) == 1:
        print('[INFO][utils.py] Store 1D tensor {}'.format(layer_name))
        Store1DBinConvert(weight, f)
      else:
        print('[INFO][utils.py] Wrong weight shape!!! '
            'Variable name is {}'.format(weight.name))

def Store4DBinConvert(weight, f):

  weight_f = weight.shape[0]
  weight_c = weight.shape[1]
  weight_h = weight.shape[2]
  weight_w = weight.shape[3]
  weight_p = 64
  # weight_p = 1
  weight_np = weight.detach().numpy()
  # if weight_c ==255:
  #   img_0 = np.zeros(weight_f, 1, weight_h, weight_w)
  #   weight_np = np.concatenate([weight_np, img_0], axis=1)
  #   weight_c=weight_c+1

  num_pixel = weight_f * weight_c * weight_h * weight_w
  step_f = weight_c * weight_h * weight_w * weight_p       # Step for filter
  step_c = weight_h * weight_w * weight_p                  # Step for col
  step_h = weight_w * weight_p                             # Step for channel
  step_w = weight_p                                        # Step for row



  data_weight = np.zeros(int(num_pixel), dtype=np.int16)
  # data_weight = np.zeros(int(num_pixel), dtype=np.float32)
  for b in range(int(weight_f / weight_p)):#比如256个filter,但是只有64个并行度，则每次只能读64个filter，那么这个b就是读完第一个64个filter之后，读第二个64个filter
    for k in range(weight_c):
      for row in range(weight_h):
        for col in range(weight_w):
          for p in range(weight_p):##每weight_p(64)个filter的第一个channal的第一行第一列，然后是第一行第二列，然后第二行，然后这个二维map遍历完之后换下一个channel。
            index = int(
                b * step_f + k * step_c + row * step_h + col * step_w + p)
            # data_weight[index] = \
                # RoundPower2(weight_np[row, col, k, b * weight_p + p])
            # data_weight[index] = weight_np[row, col, k, b * weight_p + p]
            data_weight[index] = \
                weight_np[b * weight_p + p, k, row, col]#right in order
            # if index<200:
            #   print(index,weight_np[k, b * weight_p + p, row, col],data_weight[index])
            # data_weight[index] = 1

  data_weight=np.reshape(data_weight,[-1,2])
  data_weight=data_weight[:,::-1]
  data_weight=np.reshape(data_weight,[-1])

  for npiter in np.nditer(data_weight):
    # f.write(str(npiter) + ' ')
    f.write(npiter)


def Store1DBinConvert(weight, f):
  weight_c = weight.shape[0]

  weight_np = weight.detach().numpy()
  # if weight_c ==255:
  #   a=np.zeros(1)
  #   weight_np=np.concatenate([weight_np, a], axis=0)
  #   weight_c=weight_c+1
  data_weight = np.zeros(int(weight_c), dtype=np.int32)
  # data_weight = np.zeros(int(num_pixel), dtype=np.float32)
  for b in range(int(weight_c)):
    # data_weight[b] = Round2Fixed(weight_np[b], 4, 12)
    data_weight[b] = weight_np[b]

  for npiter in np.nditer(data_weight):
    f.write(npiter)

