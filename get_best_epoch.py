import argparse
import pickle
import os
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

def get_acc(epoch_j, epoch_b, epoch_jm, epoch_bm, label, arg):
  try:
    if epoch_j != -1:
      # print('load pkl: ', os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)))
      with open(os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)), 'rb') as r1:  # j
        r1 = list(pickle.load(r1).items())  # 50816

    if epoch_b != -1:
      # print('load pkl: ', os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)))
      with open(os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)), 'rb') as r2:  # b
        r2 = list(pickle.load(r2).items())  # 50880

    if epoch_jm != -1:
      # print('load pkl: ', os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)))
      with open(os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)), 'rb') as r3:  # jm
        r3 = list(pickle.load(r3).items())  # 50880

    if epoch_bm != -1:
      # print('load pkl: ', os.path.join(work_dir + 'bm', 'epoch{}_test_score.pkl'.format(epoch_bm)))
      with open(os.path.join(work_dir + 'bm', 'epoch{}_test_score.pkl'.format(epoch_bm)), 'rb') as r4:  # bm
        r4 = list(pickle.load(r4).items())  # 50880

    right_num = total_num = right_num_5 = 0
    # for i in tqdm(range(len(label))):
    for i in range(len(label)):
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
    # print('test_num: ', total_num)
    # print('total_num: ', len(label))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    return acc, acc5
  except Exception as e:
    epoch_str = '{}_{}_{}_{}'.format(epoch_j, epoch_b, epoch_jm, epoch_bm)
    print('跳过： ', epoch_str)
    return -1,-1

def find_epoch_list(work_dir):
  # j
  epoch_j_list = []
  for epoch_j_file in os.listdir(work_dir + 'j'):
    if '.pkl' in epoch_j_file:
      epoch_j_list.append(int(epoch_j_file.split('_')[0].strip('epoch')))
  epoch_j_list = sorted(epoch_j_list)

  # b
  epoch_b_list = []
  for epoch_b_file in os.listdir(work_dir + 'b'):
    if '.pkl' in epoch_b_file:
      epoch_b_list.append(int(epoch_b_file.split('_')[0].strip('epoch')))
  epoch_b_list = sorted(epoch_b_list)

  # jm
  epoch_jm_list = []
  for epoch_jm_file in os.listdir(work_dir + 'jm'):
    if '.pkl' in epoch_jm_file:
      
      epoch_jm_list.append(int(epoch_jm_file.split('_')[0].strip('epoch')))

  epoch_jm_list = sorted(epoch_jm_list)

  # bm
  epoch_bm_list = []
  for epoch_bm_file in os.listdir(work_dir + 'bm'):
    if '.pkl' in epoch_bm_file:
      
      epoch_bm_list.append(int(epoch_bm_file.split('_')[0].strip('epoch')))

  epoch_bm_list = sorted(epoch_bm_list)

  return epoch_j_list, epoch_b_list, epoch_jm_list, epoch_bm_list

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
  epoch_jm = -1
  epoch_bm = -1

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

  # 修改的参数
  arg.dataset = 'ntu120/xsub'
  model_name = 'dev_ctr_sa1_da_fixed_aff_lsce'
  # arg.alpha = [0.6, 0.6, 0.4, 0.4]
  arg.alpha = [0.6, 0.75, 0.3, 0.15]


  work_dir = 'work_dir/' + arg.dataset + '/' + model_name + '_'
  print('epoch 开始处理')


  # 自动找pkl文件
  # epoch_j_list,epoch_b_list,epoch_jm_list,epoch_bm_list = find_epoch_list(work_dir)
  # 手动给定epoch


  epoch_j_list = [58, 63,69,77,95]
  epoch_b_list = [58,63,69,71,79,89]
  epoch_jm_list = [57,59,61,71,78,85]
  epoch_bm_list = [57,80,88,98]

  print('epoch iter time: {}m'.format(len(epoch_j_list)*len(epoch_b_list)*len(epoch_jm_list)*len(epoch_bm_list)/ 60))
  epoch_list = []
  for epoch_j in epoch_j_list:
    for epoch_b in epoch_b_list:
      for epoch_jm in epoch_jm_list:
        for epoch_bm in epoch_bm_list:
          epoch_list.append((epoch_j,epoch_b,epoch_jm,epoch_bm))

  print('epoch 处理结束')
  best_epoch = -1
  best_acc1 = -1
  epoch_acc_dict = {}
  for epoch in epoch_list:
  # for epoch in tqdm(epoch_list):
    epoch_j, epoch_b, epoch_jm, epoch_bm = epoch
    # print('j: ', epoch_j, ' b: ', epoch_b, ' jm: ', epoch_jm, ' bm: ', epoch_bm)
    acc1, acc5 = get_acc(epoch_j, epoch_b, epoch_jm, epoch_bm, label, arg)
    epoch_str = '{}_{}_{}_{}'.format(epoch_j, epoch_b, epoch_jm, epoch_bm)
    # print('epoch_str, top1: ', acc1)
    # 保留4位小数
    acc1 = round(acc1, 4)
    if acc1 > best_acc1:
      best_acc1 = acc1
      best_epoch_str = epoch_str
      
    print(epoch_str, ' top1: ', acc1, ' best ', best_epoch_str, best_acc1)

    # 保存到字典
    epoch_acc_dict[epoch_str] = acc1

sorted_epoch_acc_list = list(sorted(epoch_acc_dict.items(), key=lambda x: x[1]))
#取前5个字典值
print('arg.alpha: ', arg.alpha)
get_5_acc_list = sorted_epoch_acc_list[-5:]
for item in get_5_acc_list:
  print(item)








      # # 下面是对一次数据的对比
      # if epoch_j != -1:
      #     print('load pkl: ', os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)))
      #     with open(os.path.join(work_dir + 'j', 'epoch{}_test_score.pkl'.format(epoch_j)), 'rb') as r1:  # j
      #         r1 = list(pickle.load(r1).items())  # 50816

      # if epoch_b != -1:
      #     print('load pkl: ', os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)))
      #     with open(os.path.join(work_dir + 'b', 'epoch{}_test_score.pkl'.format(epoch_b)), 'rb') as r2:  # b
      #         r2 = list(pickle.load(r2).items())  # 50880

      # if epoch_b != -1:
      #     print('load pkl: ', os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)))
      #     with open(os.path.join(work_dir + 'jm', 'epoch{}_test_score.pkl'.format(epoch_jm)), 'rb') as r3:  # jm
      #         r3 = list(pickle.load(r3).items())  # 50880

      # if epoch_b != -1:
      #     print('load pkl: ', os.path.join(work_dir + 'bm', 'epoch{}_test_score.pkl'.format(epoch_bm)))
      #     with open(os.path.join(work_dir + 'bm', 'epoch{}_test_score.pkl'.format(epoch_bm)), 'rb') as r4:  # bm
      #         r4 = list(pickle.load(r4).items())  # 50880
      # # if arg.joint_motion_dir is not None:
      # #     with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:  # jm
      # #         r3 = list(pickle.load(r3).items())
      # # if arg.bone_motion_dir is not None:
      # #     with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:  # bm
      # #         r4 = list(pickle.load(r4).items())

      # right_num = total_num = right_num_5 = 0

      # if epoch_jm != -1 and epoch_bm != -1:
      #     # arg.alpha = [0.6, 0.6, 0.4, 0.4]
      #     arg.alpha = [0.6, 0.75, 0.3, 0.15]
      #     for i in tqdm(range(len(label))):
      #         try:
      #             # print(i)  # 50816
      #             l = label[i]
      #             _, r11 = r1[i]
      #             _, r22 = r2[i]
      #             _, r33 = r3[i]
      #             _, r44 = r4[i]
      #             r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
      #             rank_5 = r.argsort()[-5:]
      #             right_num_5 += int(int(l) in rank_5)
      #             r = np.argmax(r)
      #             right_num += int(r == int(l))
      #             total_num += 1
      #         except Exception as e:
      #             a = 1
      #             # print(i)
      #     print('test_num: ', total_num)
      #     print('total_num: ', len(label))
      #     acc = right_num / total_num
      #     acc5 = right_num_5 / total_num
      # elif epoch_jm != -1 and epoch_bm == -1:
      #     arg.alpha = [0.6, 0.6, 0.4]
      #     for i in tqdm(range(len(label))):
      #         l = label[:, i]
      #         _, r11 = r1[i]
      #         _, r22 = r2[i]
      #         _, r33 = r3[i]
      #         r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
      #         rank_5 = r.argsort()[-5:]
      #         right_num_5 += int(int(l) in rank_5)
      #         r = np.argmax(r)
      #         right_num += int(r == int(l))
      #         total_num += 1
      #     acc = right_num / total_num
      #     acc5 = right_num_5 / total_num
      # else:
      #     for i in tqdm(range(len(label))):
      #         l = label[i]
      #         _, r11 = r1[i]
      #         _, r22 = r2[i]
      #         r = r11 + r22 * arg.alpha
      #         rank_5 = r.argsort()[-5:]
      #         right_num_5 += int(int(l) in rank_5)
      #         r = np.argmax(r)
      #         right_num += int(r == int(l))
      #         total_num += 1
      #     acc = right_num / total_num
      #     acc5 = right_num_5 / total_num

      # print('model_name: ', model_name)
      # print('dataset: ', arg.dataset )
      # print('j: ', epoch_j, ' b: ', epoch_b, ' jm: ', epoch_jm, ' bm: ', epoch_bm)
      # print('arg.alpha: ', arg.alpha)
      # print('Top1 Acc: {:.4f}%'.format(acc * 100))
      # print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

