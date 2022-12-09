import torch
from darknet_19_model import Darknet_19_Model
from pathlib import Path
if __name__ == '__main__':
    model=Darknet_19_Model()
    model_dict1 = torch.load(Path('.')/'model_weights'/'quant_bn_fused_model_inference_new.pt',map_location='cpu').state_dict()
    model_dict2 = model.state_dict()
    model_list1 = list(model_dict1.keys())
    model_list2 = list(model_dict2.keys())
    len1 = len(model_list1)
    len2 = len(model_list2)
    # torch.save(model_dict2,'yolo_tiny_weights_order.pt')
    m, n = 0, 0
    layername1, layername2 = model_list1[m], model_list2[n]
    w1, w2 = model_dict1[layername1], model_dict2[layername2]
    weights_zeros=torch.zeros(64,5,3,3)
    model_dict2[layername2] = model_dict1[layername1]
    model_dict2[layername2] =torch.cat([model_dict2[layername2],weights_zeros],dim=1)

    m += 1
    n += 1
    while True:
        if  n >= len2-2:
            break
        layername1, layername2 = model_list1[m], model_list2[n]
        w1, w2 = model_dict1[layername1], model_dict2[layername2]
        # print("save one")
        # w1=w1.permute(3,2,0,1)
        if w1.shape != w2.shape:
            print('Different shape!')
            continue
        model_dict2[layername2] = model_dict1[layername1]
        m += 1
        n += 1
    layername1, layername2 = model_list1[m+2], model_list2[n]
    w1, w2 = model_dict1[layername1], model_dict2[layername2]
    weights_zeros=torch.zeros(1,1024,1,1)
    model_dict2[layername2] = model_dict1[layername1]
    model_dict2[layername2] =torch.cat([model_dict2[layername2],weights_zeros],dim=0)


    m += 1
    n += 1
    layername1, layername2 = model_list1[m+2], model_list2[n]
    w1, w2 = model_dict1[layername1], model_dict2[layername2]
    bias_zeros=torch.zeros(1)
    model_dict2[layername2] = model_dict1[layername1]
    model_dict2[layername2] =torch.cat([model_dict2[layername2],bias_zeros],dim=0)


    model.load_state_dict(model_dict2)
    torch.save(model_dict2, 'model_weights/darknet_19_weights_order_new.pt')