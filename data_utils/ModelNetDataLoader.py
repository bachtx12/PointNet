import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import glob
import h5py
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))
from data_utils import normalize_point_cloud, rotate_point_cloud, jitter_point_cloud 
class ModelNetDataset_H5PY(data.Dataset):
    def __init__(self, filelist, num_point=1024, data_augmentation=False):
        self.num_point = num_point
        self.file_list = [item.strip() for item in open(filelist).readlines()]
        self.points_list = np.zeros((1, num_point, 3))
        self.labels_list = np.zeros((1,))
        self.data_augmentation = data_augmentation
        self.num_classes = 40
        for file in self.file_list:
            data, label = self.loadh5DataFile(file)
            self.points_list = np.concatenate(
                [self.points_list, data[:, :self.num_point, :]], axis=0)
            self.labels_list = np.concatenate([self.labels_list, label.ravel()], axis=0)

        self.points_list = self.points_list[1:]
        self.labels_list = self.labels_list[1:]
        assert len(self.points_list) == len(self.labels_list)
        print('Number of Objects: ', len(self.labels_list))

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, index):
        point_set = np.copy(self.points_list[index][:, 0:3])
        point_label = self.labels_list[index].astype(np.int32)
        # point_set = normalize_point_cloud(point_set)
        if self.data_augmentation:
            point_set = rotate_point_cloud(point_set)
            point_set = jitter_point_cloud(point_set)

        return torch.from_numpy(point_set.astype(np.float32)), torch.from_numpy(np.array([point_label]).astype(np.int64))

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1024,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.cats = {}
        idx = 0
        with open(os.path.join(self.root,'modelnet_id.txt')) as f:
            for line in f:
                line = line.split('\t')
                self.cats[line[0]] = int(line[1])
        self.paths = []
        self.classes = dict(zip(sorted(self.cats), range(len(self.cats))))
        for cat in self.cats:
            self.paths +=glob.glob('%s/%s/%s/*'%(self.root,self.split, cat))

    def __getitem__(self, index):
        fn = self.paths[index]
        cls = self.cats[fn.split('/')[-2]]
        data = []
        with open(fn) as f:
            lines = f.readlines()
            for line in lines:
                data.append([float(x) for x in line.strip().split()])
        data = np.asarray(data)
        #choice = np.random.choice(len(data), self.npoints, replace=True)
        point_set = data[0:self.npoints, :]

        # point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        # point_set = point_set / dist  # scale

        # if self.data_augmentation:
        #     theta = np.random.uniform(0, np.pi * 2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        #     point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation

        #     point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls
    def __len__(self):
        return len(self.paths)
if __name__=='__main__':
    data = ModelNetDataset_H5PY('/home/ubuntu/modelnet40_ply_hdf5_2048/train.txt', data_augmentation=False)
    point, cls =data[9000]
    
    print(point)