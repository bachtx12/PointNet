import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from pointnet_utils import feature_transform_regularizer, STN3D_input, STN3D_feature
class PointNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D_input(input_channels)
        self.stn2 = STN3D_feature(64)
        self.num_classes = num_classes
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            # nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        T1 = self.stn1(x)
        x = torch.bmm(T1, x)
        x = self.mlp1(x)
        T2 = self.stn2(x)
        f = torch.bmm(T2, x)
        x = self.mlp2(f)
        x = F.max_pool1d(x, num_points).squeeze(2)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1), T1, T2
class PointNet_global(nn.Module):
    def __init__(self, input_channels, num_classes, feature_dim=128):
        super(PointNet_global, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D_input(input_channels)
        self.stn2 = STN3D_feature(64)
        self.feature_dim=feature_dim
        self.num_classes = num_classes
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            # nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes)
        )
        self.global_project = nn.Sequential(
            nn.Linear(1024, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feature_dim, bias=True)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        T1 = self.stn1(x)
        x = torch.bmm(T1, x)
        x = self.mlp1(x)
        T2 = self.stn2(x)
        f = torch.bmm(T2, x)
        x = self.mlp2(f)
        x = F.max_pool1d(x, num_points).squeeze(2)
        global_feat_proj = self.global_project(x) 
        x = self.classifier(x)
        return F.log_softmax(x, dim=1), T1, T2, global_feat_proj
def get_loss(pred, target, feat_trans, reg_weight=0.001):
    # cross entropy loss
    loss_cls = F.nll_loss(pred, target) 

    # regularize loss
    loss_reg = feature_transform_regularizer(feat_trans)

    return loss_cls + loss_reg*reg_weight


    return 0
if __name__ == '__main__':
    # print(nn.Conv1d(3,64,1).weight.size(),nn.Conv1d(3,64,1).bias.size() )
    pointnet = PointNet(3,10)
    # pointnet.train()
    data = torch.ones(4,3,1024)
    # print(pointnet(data)[0])
    with torch.no_grad():
        pointnet.eval()
        # print(pointnet)
        print(pointnet.mlp1(data)[0])