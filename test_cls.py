import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))
from data_utils import rotate_point_cloud_by_angle
from ModelNetDataLoader import ModelNetDataset, ModelNetDataset_H5PY
from ScanObjectNNDataLoader import ScanObjectNNDataset
from utils import copy_parameters
from pointnet_cls import PointNet, get_loss
import torch.nn.functional as F
from tqdm import tqdm
import json
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Classification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 32]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--model_path', type=str, required=True, help='model pre-trained')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--dataset_type', type=str, required=True, help='kind of dataset such as scanobjectnn|modelnet40|scanobjectnn10')
    parser.add_argument('--num_vote', type=int, default=1, help='kind of dataset such as scanobjectnn|modelnet40|scanobjectnn10')


    #parameter of pointnet
    return parser.parse_args()

def test():
    args = parse_args()
    try:
        os.makedirs(args.log_dir)
    except OSError:
        pass
    if args.dataset_type == 'modelnet40':
        test_dataset = ModelNetDataset_H5PY(filelist=args.dataset_path+'/test.txt', num_point=args.num_point, data_augmentation=False)
    elif args.dataset_type == 'scanobjectnn':

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path,
            split='test',
            npoints=args.num_point,
            data_augmentation=False)
    elif args.dataset_type == 'scanobjectnn10':

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path,
            split='test',
            npoints=args.num_point,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)

    print(len(test_dataset))
    # num_classes = len(dataset.classes)
    num_classes =test_dataset.num_classes
    print('classes', num_classes)

    classifier = PointNet(3, num_classes)
    
    classifier = copy_parameters(classifier,torch.load(args.model_path))

    ## Test 
    results = {}
    total_loss = 0.0
    total_point = 0.0
    total_correct = 0.0
    classifier = classifier.eval()
    classifier.cuda()
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        total_point+=points.size(0)
        target = target[:, 0].cuda()
        vote_pred = torch.zeros(points.size()[0], num_classes).cuda()
        # points = points.transpose(2, 1)
        for vid in range(args.num_vote):
            rotated_data = rotate_point_cloud_by_angle(points.cpu().numpy(), vid/float(args.num_vote) * np.pi * 2)
            points_cuda = torch.from_numpy(rotated_data).cuda()
            # print(torch.cuda.memory_allocated())
            # points_cuda = points.cuda()
            # del points_cuda
            pred, trans, trans_feat = classifier(points_cuda)
            vote_pred+=pred
        # pred_choice = pred.data.max(1)[1]
        pred_choice = vote_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct+=correct.item()
    results['Loss'] = total_loss
    results['Instance acc'] = total_correct/total_point
    with open('%s/test_results.txt'%args.log_dir, 'w') as f:
        json.dump(results, f)
if __name__=='__main__':
    test()