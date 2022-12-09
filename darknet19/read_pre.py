import torch
if __name__ == '__main__':
    #transfer the imga to 8 chs
    img_pre=torch.load('data/images/input_416.pt',map_location='cpu')
    size_1=416
    size_2=416
    img_t=img_pre['img']
    img_0 = torch.zeros(1,5, size_1, size_2)
    newimg = torch.cat([img_t, img_0], dim=1)
    torch.save(newimg,'data/images/416_img_8chs_new.pth')