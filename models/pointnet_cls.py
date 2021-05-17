import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from pointnet_utils import feature_transform_regularizer, STN3D

class PointNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
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

def get_loss(pred, target, feat_trans, reg_weight=0.001):
    # cross entropy loss
    loss_cls = F.nll_loss(pred, target) 

    # regularize loss
    loss_reg = feature_transform_regularizer(feat_trans)

    return loss_cls + loss_reg*reg_weight


    return 0
if __name__ == '__main__':
    print(nn.Conv1d(3,64,1).weight.size(),nn.Conv1d(3,64,1).bias.size() )
    pointnet = PointNet(3,40)
    data = torch.rand(4,1024,3)
    # print(pointnet(data))