import torch
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from ModelNetDataLoader import ModelNetDataset, ModelNetDataset_H5PY
from ScanObjectNNDataLoader import ScanObjectNNDataset
from utils import copy_parameters
from matplotlib import pyplot
from pointnet_cls import PointNet
from sklearn.manifold import TSNE
import numpy as np
from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
def tSNE(z,t, id_cat, file_out):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    fig, axes= pyplot.subplots()
    tsne = TSNE(perplexity=15)
    z = tsne.fit_transform(z)
    kmeans = KMeans(n_clusters=10).fit(z)
    nmi = normalized_mutual_info_score(t,kmeans.labels_)
    p = purity_score(t,kmeans.labels_)
    axes.set_title('NMI %s Purity %s'%(nmi, p))
    classes = list(id_cat.keys())
    for i in range(len(classes)):
        ids = [k for k in range(len(t)) if t[k] == classes[i]] 
        axes.scatter(z[ids,0], z[ids,1], c=colors[i], label=id_cat[classes[i]])
    axes.legend(loc='center right')
    pyplot.savefig(file_out)
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('SVM classification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 32]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--feature_transform', action='store_true', help='Using feature transform in pointnet')
    parser.add_argument('--model_path', type=str, required=True, help='model pre-trained')
    parser.add_argument('--file_image', type=str, required=True, help='file image saved')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--dataset_type', type=str, required=True, help='kind of dataset such as scanobjectnn|modelnet40|scanobjectnn10')
    parser.add_argument('--data_aug', action='store_true', help='Using data augmentation for training phase')
    #parameter of pointnet
    return parser.parse_args()

def train():
    args = parse_args()
    print(args)
    if args.dataset_type == 'modelnet40':
        test_dataset = ModelNetDataset(
            root=args.dataset_path,
            npoints=args.num_point,
            split='test',
            tsne = [2,5,7,10,15,20,21,30, 32,33],
            data_augmentation=False        
            )
    elif args.dataset_type == 'scanobjectnn':

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path,
            split='test',
            npoints=args.num_point,
            data_augmentation=False)
    elif args.dataset_type == 'scanobjectnnbg':

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
            num_workers=8)

    print( len(test_dataset))
    # num_classes = len(dataset.classes)
    num_classes =test_dataset.num_classes
    classifier = PointNet(input_channels=3, num_classes=num_classes,global_feature=True, feature_transform=args.feature_transform)
    if args.model_path != '':
        classifier = copy_parameters(classifier,torch.load(args.model_path))
    classifier.cuda()
    X_test = []
    Y_test = []
    with torch.no_grad():
        classifier.eval()

        for points, target in tqdm(testdataloader, total=len(testdataloader), smoothing=0.9):
            target = target
            points, target = points.cuda(), target.long().cuda()


            global_feature = classifier(points)

            X_test.append(global_feature.cpu().numpy())
            Y_test.append(target.cpu().numpy())
    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    tSNE(X_test, Y_test.flatten(),test_dataset.id_cat, args.file_image)
if __name__ == '__main__':
    train()