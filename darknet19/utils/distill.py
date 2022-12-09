import torch
import torch.nn as nn
import pdb

def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou
def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]

    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    # pdb.set_trace()
    for i in range(len(model.model)):
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)

            use_best_anchor = False
            if use_best_anchor:
                iou, a = iou.max(0)  # best iou and anchor
            else:  # use all anchors
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
                iou = iou.view(-1)  # use all ious

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            reject = True
            if reject:
                j = iou > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # GIoU
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() <= model.nc, 'Target classes exceed model classes'

    return tcls, tbox, indices, av


def distillation_loss1(output_s, output_t, num_classes, batch_size):
    T = 3.0
    Lambda_ST = 0.001
    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    output_s = torch.cat([i.view(-1, num_classes + 5) for i in output_s])
    output_t = torch.cat([i.view(-1, num_classes + 5) for i in output_t])
    loss_st  = criterion_st(nn.functional.log_softmax(output_s/T, dim=1), nn.functional.softmax(output_t/T,dim=1))* (T*T) / batch_size
    return loss_st * Lambda_ST