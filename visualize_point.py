import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
import sys
import scipy.misc
import imageio
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))
from data_utils import rotate_point_cloud_by_angle
from pc_utils import *
from utils import copy_parameters
from ModelNetDataLoader import ModelNetDataset, ModelNetDataset_H5PY
from ScanObjectNNDataLoader import ScanObjectNNDataset
from pointnet_cls import PointNet_critical
import torch.nn.functional as F
from tqdm import tqdm
import json
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Classification')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training [default: 32]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--feature_transform', type=bool, default=True, help='Using feature transform in pointnet')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--model_path', type=str, required=True, help='model pre-trained')
    parser.add_argument('--test_class', type=str, default='all', help='class visualize')
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
        test_dataset = ModelNetDataset(
            root=args.dataset_path,
            npoints=args.num_point,
            split='train',
            test_class=args.test_class,
            data_augmentation=False        
            )
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
            shuffle=False,
            num_workers=4)

    print(len(test_dataset))
    # num_classes = len(dataset.classes)
    num_classes =test_dataset.num_classes
    print('classes', num_classes)

    classifier = PointNet_critical(3, args.feature_transform)
    
    classifier = copy_parameters(classifier,torch.load(args.model_path))
    # classifier.load_state_dict(torch.load(args.model_path), strict=False)

    ## Test 
    with torch.no_grad():
        classifier = classifier.eval()
        classifier.cuda()
        for i, data in tqdm(enumerate(testdataloader, 0)):
            if i==0:
                continue
            points, target = data
            target = target[:, 0].cuda()
            # points = points.transpose(2, 1)
            orgin_data = points.squeeze().numpy().copy()
            global_feature = None
            critical_index = None
            for vid in range(args.num_vote):
                rotated_data = rotate_point_cloud_by_angle(points.cpu().numpy(), vid/float(args.num_vote) * np.pi * 2)
                points_cuda = torch.from_numpy(rotated_data).cuda()

                global_feature, critical_index, _ = classifier(points_cuda)
                critical_index = critical_index.squeeze().cpu().numpy()
            critical_points = []
            for index in critical_index:
                critical_points.append(orgin_data[int(index), :].tolist())
            critical_points = list(set([tuple(t) for t in critical_points]))
            # print(len(critical_index), len(critical_points))
            print('global feat: ', global_feature)
            img_filename = '%s/critical_points.jpg'%(args.log_dir) 
            # output_img = point_cloud_three_views(np.squeeze( critical_points ))
            output_img = draw_point_cloud(np.squeeze( critical_points ))
            imageio.imwrite(img_filename, output_img)
            print('original: ', orgin_data)
            img_filename = '%s/original.jpg'%(args.log_dir)
            output_img = draw_point_cloud(np.squeeze( orgin_data ))
            imageio.imwrite(img_filename, output_img)

            max_position = [-1,-1,-1]
            min_position = [1, 1, 1]

            for point_index in range(len(orgin_data)):
                max_position = np.maximum(max_position, orgin_data[point_index,:])
                min_position = np.minimum(min_position, orgin_data[point_index,:])
            print(max_position, min_position)
            upper_bound = orgin_data.copy().tolist()
            search_step = 0.02
            for x in np.linspace(min_position[0], max_position[0], int((max_position[0]-min_position[0])/search_step) +1):
                    for y in np.linspace(min_position[1], max_position[1], int((max_position[1]-min_position[1])//search_step) +1):
                        for z in np.linspace(min_position[2], max_position[2], int((max_position[2]-min_position[2])//search_step) +1):
                            upper_bound.append([x,y,z])
            print(len(upper_bound))
            upper_torch = torch.from_numpy(np.asarray(upper_bound).astype(np.float32)).unsqueeze(0)
            upper_torch = upper_torch.cuda()
            _, _, upper_feat = classifier(upper_torch)
            upper_feat = upper_feat.squeeze().cpu().numpy().T
            print(upper_feat.shape)

            global_feature = global_feature.squeeze().cpu().numpy()
            upper = orgin_data.copy().tolist()
            upper_index = []
            for i in range(len(upper_feat)):
                if (upper_feat[i]<global_feature).all():
                    upper.append(upper_bound[i])
        
            upper = np.array(upper)
            print('Upper: ',upper)
            img_filename = '%s/upper_bound.jpg'%(args.log_dir)
            output_img = draw_point_cloud(np.squeeze( np.asarray(upper) ))
            imageio.imwrite(img_filename, output_img)
            break
if __name__=='__main__':
    test()