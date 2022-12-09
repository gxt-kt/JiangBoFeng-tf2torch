# Loss functions

import torch
import torch.nn as nn
import pdb
from utils.general import bbox_iou
from utils.torch_utils import is_parallel
import torch.nn.functional as F
from config import opt
# p:predict    t_p: teacher_predict   mode：学生网络
def compute_distillation_output_loss(p, t_p, model, dist_loss="l2", T=20, reg_norm=None):
    t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError(
            "reduction must be mean in distillation mode!")
    #box损失函数
    DboxLoss = nn.MSELoss(reduction="none")
    #class 损失函数         不同的损失函数 
    if dist_loss == "l2":
        DclsLoss = nn.MSELoss(reduction="none")
    elif dist_loss == "kl":
        DclsLoss = nn.KLDivLoss(reduction="none")
    else:
        DclsLoss = nn.BCEWithLogitsLoss(reduction="none")
    #obj损失函数
    DobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions 
        #len(p) =2 或者3  与最后的detact数量有关      p[1].shape = ([16, 3, 40, 40, 85])

        t_pi = t_p[i]  #和teacher相对应
        # t_pi = t_p[i+1]# yolov3 最后detect有三个 要和tiny对应 
        #t_pi[..., 4]取除最后一纬 例如[16, 3, 40, 40]      第四个   sigmoid 激活函数
          #找teacher
        t_obj_scale = t_pi[..., 4].sigmoid()    #size(16,3,40,40)
        # BBox
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)   #unsqueeze(-1)之后（16，3，40，40，1）    repeat(1,1,1,1,4)之后   （16，3，40，40，4）
       # Dbox 
        if not reg_norm:
            t_lbox += torch.mean(DboxLoss(pi[..., :4],
                                          t_pi[..., :4]) * b_obj_scale)
        else:
            wh_norm_scale = reg_norm[i].unsqueeze(
                0).unsqueeze(-2).unsqueeze(-2)
            t_lbox += torch.mean(DboxLoss(pi[..., :2].sigmoid(),
                                          t_pi[..., :2].sigmoid()) * b_obj_scale)
            t_lbox += torch.mean(DboxLoss(pi[..., 2:4].sigmoid(),
                                          t_pi[..., 2:4].sigmoid() * wh_norm_scale) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1,
                                                           1, 1, 1, model.nc)
            if dist_loss == "kl":
                kl_loss = DclsLoss(F.log_softmax(pi[..., 5:]/T, dim=-1),
                                   F.softmax(t_pi[..., 5:]/T, dim=-1)) * (T * T)
                t_lcls += torch.mean(kl_loss * c_obj_scale)
            else:

                t_lcls += torch.mean(DclsLoss(pi[..., 5:].repeat(1,1,1,1,4),t_pi[..., 5:]) * c_obj_scale)

        t_lobj += torch.mean(DobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    # pdb.set_trace()
    t_lbox *= h['box'] * h['dist']
    t_lobj *= h['obj'] * h['dist']
    t_lcls *= h['cls'] * h['dist']
    bs = p[0].shape[0]  # batch size
    dloss = (t_lobj + t_lbox + t_lcls) * bs
    return dloss



def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            #FocalLoss 里面就是BCEWithLogitsLoss gamma =1.5 
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        if opt.prune:
            det = model.head_self  #剪枝过后
        else:
            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        loss = lbox + lobj + lcls

        return loss* bs, torch.cat((lbox, lobj, lcls,loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeDstillLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, distill_ratio=0.5, temperature=10):
        super(ComputeDstillLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.distill_ratio = distill_ratio
        self.T = temperature
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.L2Logits = nn.MSELoss()
        self.KLDistillLoss = nn.KLDivLoss()
        # self.BCEDistillLoss = nn.BCEWithLogitsLoss()
        # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Detect() module
        
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(
            16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def KlSoftmaxLoss(self, student_var, teacher_var):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        T = self.T
        KD_loss = self.KLDistillLoss(F.log_softmax(student_var/T, dim=-1),
                                     F.softmax(teacher_var/T, dim=-1)) * (T * T)

        return KD_loss

    # predictions, targets, model
    def __call__(self, p, targets, soft_loss='kl'):
        device = targets.device
        dloss = torch.zeros(1, device=device)
        tcls, tbox, tlogits, indices, anchors = self.build_targets(
            p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # prediction subset corresponding to targets
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                dloss += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (
                    1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    dloss += self.BCEcls(ps[:, 5:], t)  # BCE
                    if soft_loss == 'kl':
                        dloss += self.KlSoftmaxLoss(ps[:, 5:], tlogits[i])
                    elif soft_loss == 'l2':
                        dloss += self.L2Logits(ps[:, 5:], tlogits[i])

            obji = self.BCEobj(pi[..., 4], tobj)
            dloss += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * \
                    0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        dloss *= self.distill_ratio
        bs = tobj.shape[0]  # batch size
        return dloss * bs

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        nc = targets.shape[1] - 6  # number of classes
        # targets.shape = (16, 6+20)
        tcls, tbox, indices, tlogits, anch = [], [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7 + nc, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(
            na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # 一共三层
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # t.shape = (3, 16, 6+20+1)
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio (3, 16, 2)
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare (3, 16)
                t = t[j]  # 表示这一层匹配到的anchor

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            logits = t[:, 6:6+nc]
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices
            # image, anchor, grid indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tlogits.append(logits)

        return tcls, tbox, tlogits, indices, anch

# def wh_iou(box1, box2):
#     # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
#     box2 = box2.t()

#     # w, h = box1
#     w1, h1 = box1[0], box1[1]
#     w2, h2 = box2[0], box2[1]

#     # Intersection area
#     inter_area = torch.min(w1, w2) * torch.min(h1, h2)

#     # Union Area
#     union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

#     return inter_area / union_area  # iou
# def build_targets(model, targets):
#     # targets = [image, class, x, y, w, h]

#     nt = len(targets)
#     tcls, tbox, indices, av = [], [], [], []
#     multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
#     for i in model.yolo_layers:
#         # get number of grid points and anchor vec for this yolo layer
#         if multi_gpu:
#             ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
#         else:
#             ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

#         # iou of targets-anchors
#         t, a = targets, []
#         gwh = t[:, 4:6] * ng
#         if nt:
#             iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)

#             use_best_anchor = False
#             if use_best_anchor:
#                 iou, a = iou.max(0)  # best iou and anchor
#             else:  # use all anchors
#                 na = len(anchor_vec)  # number of anchors
#                 a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
#                 t = targets.repeat([na, 1])
#                 gwh = gwh.repeat([na, 1])
#                 iou = iou.view(-1)  # use all ious

#             # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
#             reject = True
#             if reject:
#                 j = iou > model.hyp['iou_t']  # iou threshold hyperparameter
#                 t, a, gwh = t[j], a[j], gwh[j]

#         # Indices
#         b, c = t[:, :2].long().t()  # target image, class
#         gxy = t[:, 2:4] * ng  # grid x, y
#         gi, gj = gxy.long().t()  # grid x, y indices
#         indices.append((b, a, gj, gi))

#         # GIoU
#         gxy -= gxy.floor()  # xy
#         tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
#         av.append(anchor_vec[a])  # anchor vec

#         # Class
#         tcls.append(c)
#         if c.shape[0]:  # if any targets
#             assert c.max() <= model.nc, 'Target classes exceed model classes'

#     return tcls, tbox, indices, av

# def distillation_loss2(model, targets, output_s, output_t):
#     reg_m = 0.0
#     T = 3.0
#     Lambda_cls, Lambda_box = 0.0001, 0.001

#     criterion_st = torch.nn.KLDivLoss(reduction='sum')
#     ft = torch.cuda.FloatTensor if output_s[0].is_cuda else torch.Tensor
#     lcls, lbox = ft([0]), ft([0])

#     tcls, tbox, indices, anchor_vec = build_targets(model, targets)
#     reg_ratio, reg_num, reg_nb = 0, 0, 0
#     for i, (ps, pt) in enumerate(zip(output_s, output_t)):  # layer index, layer predictions
#         b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

#         nb = len(b)
#         if nb:  # number of targets
#             pss = ps[b, a, gj, gi]  # prediction subset corresponding to targets
#             pts = pt[b, a, gj, gi]

#             psxy = torch.sigmoid(pss[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
#             psbox = torch.cat((psxy, torch.exp(pss[:, 2:4]) * anchor_vec[i]), 1).view(-1, 4)  # predicted box

#             ptxy = torch.sigmoid(pts[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
#             ptbox = torch.cat((ptxy, torch.exp(pts[:, 2:4]) * anchor_vec[i]), 1).view(-1, 4)  # predicted box


#             l2_dis_s = (psbox - tbox[i]).pow(2).sum(1)
#             l2_dis_s_m = l2_dis_s + reg_m
#             l2_dis_t = (ptbox - tbox[i]).pow(2).sum(1)
#             l2_num = l2_dis_s_m > l2_dis_t
#             lbox += l2_dis_s[l2_num].sum()
#             reg_num += l2_num.sum().item()
#             reg_nb += nb

#         output_s_i = ps[..., 4:].view(-1, model.nc + 1)
#         output_t_i = pt[..., 4:].view(-1, model.nc + 1)
#         lcls += criterion_st(nn.functional.log_softmax(output_s_i/T, dim=1), nn.functional.softmax(output_t_i/T,dim=1))* (T*T) / ps.size(0)

#     if reg_nb:
#         reg_ratio = reg_num / reg_nb

#     return lcls * Lambda_cls + lbox * Lambda_box, reg_ratio