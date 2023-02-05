import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        # required=True,
                        default = 'ntu120/xsub',
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    arg = parser.parse_args()
    # 默认参数
    epoch_j = -1
    epoch_b = -1
    # epoch_b = 0
    epoch_jm = -1
    epoch_bm = -1


    # 修改的参数

    arg.dataset = 'ntu120/xsub'


    model_name = 'sectr'

    work_dir = 'work_dir/' + arg.dataset + '/' + model_name + '_'
    
    epoch_j = 61
    epoch_b = 94

    # epoch_b = 0
    epoch_jm = 92
    epoch_bm = 71


    """
j
[ Thu Jan 26 01:49:12 2023 ] Best accuracy: 0.8442216981132076
[ Thu Jan 26 01:49:12 2023 ] Epoch number: 61

b
[ Sat Jan 28 20:40:04 2023 ] Best accuracy: 0.8557389937106918  85.57%
[ Sat Jan 28 20:40:04 2023 ] Epoch number: 94

jm

[ Mon Jan 30 23:11:04 2023 ] best_epoch: 92 best_acc: 80.73%
[ Mon Jan 30 23:21:18 2023 ] Best accuracy: 0.8072523584905661

bm

[ Thu Feb  2 10:42:18 2023 ] best_epoch: 71 best_acc: 80.60%

test_num:  50880
total_num:  50919
model_name:  sectr
dataset:  ntu120/xsub
j:  61  b:  94  jm:  92  bm:  71
Top1 Acc: 88.4277%
Top5 Acc: 98.2410%

    """

    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]  # 50919
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError


    
    if epoch_j != -1:
        print('load pkl: ', os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)))
        with open(os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)), 'rb') as r1:  # j
            r1 = list(pickle.load(r1).items())  # 50816

    if epoch_b != -1:
        print('load pkl: ', os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)))
        with open(os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)), 'rb') as r2:  # b
            r2 = list(pickle.load(r2).items())  # 50880

    if epoch_b != -1:
        print('load pkl: ', os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)))
        with open(os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)), 'rb') as r3:  # jm
            r3 = list(pickle.load(r3).items())  # 50880

    if epoch_b != -1:
        print('load pkl: ', os.path.join(work_dir + 'bm', 'epoch{}_test_score.pkl'.format(epoch_bm)))
        with open(os.path.join(work_dir + 'bm', 'epoch{}_test_score.pkl'.format(epoch_bm)), 'rb') as r4:  # bm
            r4 = list(pickle.load(r4).items())  # 50880
    # if arg.joint_motion_dir is not None:
    #     with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:  # jm
    #         r3 = list(pickle.load(r3).items())
    # if arg.bone_motion_dir is not None:
    #     with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:  # bm
    #         r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    if epoch_jm != -1 and epoch_bm != -1:
        arg.alpha = [0.6, 0.6, 0.4, 0.4]
        for i in tqdm(range(len(label))):
            try:
                # print(i)  # 50816
                l = label[i]
                _, r11 = r1[i]
                _, r22 = r2[i]
                _, r33 = r3[i]
                _, r44 = r4[i]
                r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
                rank_5 = r.argsort()[-5:]
                right_num_5 += int(int(l) in rank_5)
                r = np.argmax(r)
                right_num += int(r == int(l))
                total_num += 1
            except Exception as e:
                a = 1
                # print(i)
        print('test_num: ', total_num)
        print('total_num: ', len(label))
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif epoch_jm != -1 and epoch_bm == -1:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('model_name: ', model_name)
    print('dataset: ', arg.dataset )
    print('j: ', epoch_j, ' b: ', epoch_b, ' jm: ', epoch_jm, ' bm: ', epoch_bm)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
