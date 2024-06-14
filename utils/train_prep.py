import torch
import numpy as np
import random
import torch.nn as nn

def _seed_all(random_seed=1226):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)

def set_trainable(model, boolean: bool = True, except_layers: list = [], device_ids: list = []):
    if boolean:
        for i, param in model.named_parameters():
            param.requires_grad = True
        if len(except_layers) > 0:
            for layer in except_layers:
                assert layer is not None
                if len(device_ids) <= 1:
                    for param in getattr(model, layer).parameters():
                        param.requires_grad = False
                else:
                    for param in getattr(model.module, layer).parameters():
                        param.requires_grad = False
    else:
        for i, param in model.named_parameters():
            param.requires_grad = False
        if len(except_layers) > 0:
            for layer in except_layers:
                assert layer is not None
                if len(device_ids) <= 1:
                    for param in getattr(model, layer).parameters():
                        param.requires_grad = True
                else:
                    for param in getattr(model.module, layer).parameters():
                        param.requires_grad = True
    return model

      
def binary_func_sep(logits, label, UUID,
                    ce_loss_record_0=None, ce_loss_record_1=None, ce_loss_record_2=None,
                    acc_record_0=None, acc_record_1=None, acc_record_2=None, return_sum=True, n_sources=3):
    
    label = label.float()
    correct_0, correct_1, correct_2 = 0, 0, 0
    total_0, total_1, total_2 = 1, 1, 1
    
    label = label.to(torch.long)
    ce_loss = nn.CrossEntropyLoss().cuda()
    
    # loss each domain
    indx_0 = (UUID == 0).cpu()
    if indx_0.sum().item() > 0:
        logit_0 = logits[indx_0].squeeze()
        cls_loss_0 = ce_loss(logit_0, label[indx_0])
        predicted_0 = (logit_0[:,1] > 0.5).float()
        total_0 += len(logit_0)
        correct_0 += predicted_0.cpu().eq(label[indx_0].cpu()).sum().item()
    else:
        logit_0 = []
        cls_loss_0 = torch.zeros(1).cuda()

    indx_1 = (UUID == 1).cpu()
    if indx_1.sum().item() > 0:
        logit_1 =  logits[indx_1].squeeze()
        cls_loss_1 = ce_loss(logit_1, label[indx_1])
        predicted_1 = (logit_1[:,1] > 0.5).float()
        total_1 += len(logit_1)
        correct_1 += predicted_1.cpu().eq(label[indx_1].cpu()).sum().item()
    else:
        logit_1 = []
        cls_loss_1 = torch.zeros(1).cuda()

    indx_2 = (UUID == 2).cpu()
    if indx_2.sum().item() > 0:
        logit_2 =  logits[indx_2].squeeze()
        cls_loss_2 = ce_loss(logit_2, label[indx_2])
        predicted_2 = (logit_2[:,1] > 0.5).float()
        total_2 += len(logit_2)
        correct_2 += predicted_2.cpu().eq(label[indx_2].cpu()).sum().item()
    else:
        logit_2 = []
        cls_loss_2 = torch.zeros(1).cuda()

    if ce_loss_record_0 is not None:
        ce_loss_record_0.update(cls_loss_0.data.item(), len(logit_0))
        ce_loss_record_1.update(cls_loss_1.data.item(), len(logit_1))
        ce_loss_record_2.update(cls_loss_2.data.item(), len(logit_2))
        acc_record_0.update(correct_0/total_0, total_0)
        acc_record_1.update(correct_1/total_1, total_1)
        acc_record_2.update(correct_2/total_2, total_2)

    if return_sum: 
        return (cls_loss_0 + cls_loss_1 + cls_loss_2)/n_sources
    return [cls_loss_0/n_sources, cls_loss_1/n_sources, cls_loss_2/n_sources][:n_sources]
