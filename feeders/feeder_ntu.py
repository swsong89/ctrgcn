import os
import numpy as np

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        # mmap_mode 相对于直接读入训练时间增加了1/3, 之前17m,现在25m
        mmap_mode = False
        print('mmap_mode: ', mmap_mode)
        if not mmap_mode:
            npz_data = np.load(self.data_path, mmap_mode='r')
            # npz_data = np.load(self.data_path)
            print('ntu data loading')
            if self.split == 'train':
                self.data = npz_data['x_train']
                self.label = np.where(npz_data['y_train'] > 0)[1]
                self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            elif self.split == 'test':
                self.data = npz_data['x_test']
                self.label = np.where(npz_data['y_test'] > 0)[1]
                self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            else:
                raise NotImplementedError('data split only supports train/test')
            N, T, _ = self.data.shape
            self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        else:
            data_path_name = self.data_path.replace('.npz', '') #'./data/ntu/NTU60_CV'
            print('data_path_name: ', data_path_name)
            train_data_path = data_path_name + '_train_data.npy'
            if not os.path.exists(train_data_path):  # '_train_data.npz' 不存在还没有生成
                print('train_data_path is not exist: ', train_data_path)
                print('data convert loading: ', self.data_path)
                npz_data = np.load(self.data_path)
                data = npz_data['x_train']
                N, T, _ = data.shape  # N,T,150
                print('train_data shape: ', data.shape)
                data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N,C,T,V,M, N,3,T,25,2
                label = np.where(npz_data['y_train'] > 0)[1]
                np.save(data_path_name + '_train_data', data)
                np.save(data_path_name + '_train_label', label)
                data = npz_data['x_test']
                N, T, _ = data.shape  # N,T,150
                print('test_data shape: ', data.shape)
                data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N,C,T,V,M, N,3,T,25,2
                label = np.where(npz_data['y_test'] > 0)[1]
                np.save(data_path_name + '_val_data', data)
                np.save(data_path_name + '_val_label', label)
                print('data convert end')
                print('ntu ' + self.split + ' data loading')

            if self.split == 'train':
                self.data = np.load(data_path_name + '_train_data.npy', mmap_mode='r')
                self.label = np.load(data_path_name + '_train_label.npy')
                self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            elif self.split == 'test':
                self.data = np.load(data_path_name + '_val_data.npy', mmap_mode='r')
                self.label = np.load(data_path_name + '_val_label.npy')
                self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            else:
                raise NotImplementedError('data split only supports train/test')

        print('ntu ' + self.split + ' data end')

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()  # score从小到大排列后，输出其索引值
        newlabel = self.label[0:len(rank)]
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(newlabel)]  # i：索引值，l：value值
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
