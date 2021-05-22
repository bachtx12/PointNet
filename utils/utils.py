#Ref https://github.com/AnTao97/dgcnn.pytorch/blob/master/util.py
#Ref https://github.com/hansen7/OcCo/blob/master/OcCo_Torch/utils/Torch_Utility.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def copy_parameters(model, pretrained, verbose=True):
    feat_dict = model.state_dict()
    #load pre_trained self-supervised
    pretrained_dict = pretrained
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                    k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()}
    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    feat_dict.update(pretrained_dict)
    model.load_state_dict(feat_dict)
    return model
def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
        m.eps =1e-3
        # print(m)
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
        # print(m)
def init_zeros(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.bias, 1e-10)
        torch.nn.init.zeros_(m.weight)
    else:
        print('Wrong layer TNet')
        exit()
        # print(m)