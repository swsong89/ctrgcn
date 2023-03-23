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

    arg.dataset = 'kinetics/xsub'


    model_name = 'dev_ctr_sa1_da_aff_lsce_w150'

    work_dir = 'work_dir/' + arg.dataset + '/' + model_name + '_'
    
    epoch_j = 41
    epoch_b = 45

    # epoch_b = 0
    epoch_jm = -1
    epoch_bm = -1


    """
j
[ Tue Mar 21 09:47:58 2023 ] --------------------best epoch acc: 30  27.81%
[ Tue Mar 21 21:14:13 2023 ] --------------------best epoch acc: 41  31.04%

b
[ Mon Mar 20 08:16:57 2023 ] --------------------best epoch acc: 45  29.99%

jm
                                                            ctr  73  80.78%
[ Sat Mar  4 08:50:55 2023 ] --------------------best epoch acc: 65  83.81%

bm
                                                             ctr 69  80.95%
[ Fri Mar  3 15:19:42 2023 ] --------------------best epoch acc: 98  83.53%
model_name:  dev_ctr_sa1_da_aff_lsce_w150
dataset:  kinetics/xsub  19796
j:  24  b:  45  jm:  -1  bm:  -1
arg.alpha:  1
Top1 Acc: 31.0921%
Top5 Acc: 52.0812%
j:  41  b:  45  jm:  -1  bm:  -1
arg.alpha:  1
Top1 Acc: 32.4611%
Top5 Acc: 53.5361%
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
    elif 'kinetic' in arg.dataset:
        npz_data = np.load('./data/' + 'kinetics/kinetics-skeleton/' + 'val_data.npy', mmap_mode='r')
        kinetic_dict = np.load('./data/' + 'kinetics/kinetics-skeleton/' + 'val_label.pkl',allow_pickle=True)
        label = np.array(list(kinetic_dict)[1])
    else:
        raise NotImplementedError
    print('begin')
    unvalid_label_num = 0
    for i in range(len(npz_data)):

        valid_frame_num = np.sum(npz_data[i].sum(0).sum(-1).sum(-1) != 0)  # 统计这个样本数据不为0的帧数
        if valid_frame_num == 0:
            unvalid_label_num += 1
            print('index, frame, label: ', i, valid_frame_num, label[i]) 
    print('label_sum: ', len(npz_data), ' unvalid_label_num: ', unvalid_label_num)
    print('end')

    # print('begin')
    # unvalid_label_num = 0
    # npz_train_data = np.load('./data/' + 'kinetics/kinetics-skeleton/' + 'train_data.npy', mmap_mode='r')

    # for i in range(len(npz_train_data)):

    #     valid_frame_num = np.sum(npz_train_data[i].sum(0).sum(-1).sum(-1) != 0)  # 统计这个样本数据不为0的帧数
    #     if valid_frame_num == 0:
    #         unvalid_label_num += 1
    #         print('index, frame, label: ', i, valid_frame_num, 0) 
    # print('label_sum: ', len(npz_train_data), ' unvalid_label_num: ', unvalid_label_num)
    # print('end')    label_sum:  240436  unvalid_label_num:  1328
    if epoch_j != -1:
        print('load pkl: ', os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)))
        with open(os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)), 'rb') as r1:  # j
            r1 = list(pickle.load(r1).items())  # 50816

    if epoch_b != -1:
        print('load pkl: ', os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)))
        with open(os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)), 'rb') as r2:  # b
            r2 = list(pickle.load(r2).items())  # 50880

    if epoch_jm != -1:
        print('load pkl: ', os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)))
        with open(os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)), 'rb') as r3:  # jm
            r3 = list(pickle.load(r3).items())  # 50880

    if epoch_bm != -1:
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
        # arg.alpha = [0.6, 0.6, 0.4, 0.4]
        arg.alpha = [0.6, 0.75, 0.3, 0.15]
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
    print('arg.alpha: ', arg.alpha)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
