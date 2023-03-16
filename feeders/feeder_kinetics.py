import numpy as np
import random
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
        # npz_data = np.load(self.data_path)
        print('kinetics data_path: ', self.data_path)
        if self.split == 'train':
            print('kinetics train data loading')
            self.data = np.load(self.data_path + '/train_data.npy', mmap_mode='r')  # (19796, 3, 300, 18, 2)  N，C，T，V，M
            print('kinetics train data end')
            kinetic_dict = np.load(self.data_path + '/train_label.pkl', allow_pickle=True) # [0:json, 1:list] 19796 [name,index]
            self.label = np.array(list(kinetic_dict)[1])
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            print('kinetics train data loading')
            self.data = np.load(self.data_path + '/val_data.npy', mmap_mode='r')   # (19796, 3, 300, 18, 2)  N，C，T，V，M
            print('kinetics train data end')
            kinetic_dict = np.load(self.data_path + '/val_label.pkl', allow_pickle=True) # [0:json, 1:list] 19796 [name,index]
            self.label = np.array(list(kinetic_dict)[1])
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            # kinetic_sample_name = ['test_' + str(i) for i in range(len(kinetic_data))]
            # self.data = npz_data['x_test']
            # self.label = np.where(npz_data['y_test'] > 0)[1]
            # self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        self.data_nums = len(self.label)
        # N, T, _ = self.data.shape
        # self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

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
        # 存在部分帧的数据为0，比如index 173578 self.data[173578]全是0
        is_not_valid = True
        while is_not_valid: 
            try: 
                data_numpy = self.data[index]
                label = self.label[index]
                data_numpy = np.array(data_numpy)  # [3,300,18,2] C,T,V,M
                valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)  # 加C,M,V剩下T,统计不为0的帧数
                # reshape Tx(MVC) to CTVM
                data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
                if self.random_rot:
                    data_numpy = tools.random_rot(data_numpy)
                if self.bone:
                    from .bone_pairs import kinetics_pairs
                    bone_data_numpy = np.zeros_like(data_numpy)
                    for v1, v2 in kinetics_pairs:
                        bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
                    data_numpy = bone_data_numpy
                if self.vel:
                    data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
                    data_numpy[:, -1] = 0
                is_not_valid = False
            except Exception as e:
                # print('index: ', index, ' frame: ', valid_frame_num)  # 打印 无效数据帧对应的 index
                index = random.randint(0, self.data_nums)
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
if __name__ == '__main__':
    # 123
    data_path = './data/kinetics/kinetics-skeleton/val_data.npy'  # 
    kinetic_data = np.load(data_path)  # (19796, 3, 300, 18, 2)  N，C，T，V，M
    data_path = './data/kinetics/kinetics-skeleton/val_label.pkl' 
    kinetic_dict = np.load(data_path, allow_pickle=True) # [0:json, 1:list] 19796 [name,index]
    kinetic_label = np.array(list(kinetic_dict)[1])
    kinetic_sample_name = ['test_' + str(i) for i in range(len(kinetic_data))]
    data_path = 'data/ntu/NTU60_CS.npz'
    ntu = np.load(data_path) #  ['x_train', 'y_train', 'x_test', 'y_test']
    ntu_x_test = ntu['x_test']  # (16487, 300, 150)  150 = 3*25*2  N,T, MVC 最后要变成N,C,T，V，M
    ntu_x_label = np.where(ntu['y_test'] > 0)[1]
    ntu_sample_name = ['test_' + str(i) for i in range(len(ntu_x_test))]

    print('123')
