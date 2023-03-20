# import torch
# import torch.nn.functional as F
# import torch.nn as nn
import numpy as np

file_path = 'config/txt/ntu120/' + '/1_ntu120_xsub_dev_ctr_sa1_da_fixed_aff_lsce_b.txt'
file_path = 'config/txt/ntu120/' + '/1_ntu120_xsub_ctr_b.txt'

with open(file_path, "r") as f:  # 打开文件
    data = f.readlines()  # 读取文件
    # print(len(data))
    epoch = -1
    acc = -1
    train_epoch_list = []
    train_acc_list = []

    eval_epoch_list = []
    eval_top1_list = []
    eval_top5_list = []
    for line in data:
      if 'Training epoch:' in line:
        line_sep = line.strip().split(' ')
        epoch = line_sep[-1]
        train_epoch_list.append(epoch)
      if 'Mean training acc:' in line:
        line_sep = line.strip().split(' ')
        acc = line_sep[-1].split('%')[0]
        # print(epoch, acc)
        train_acc_list.append(acc)


      if 'Eval epoch:' in line:
        line_sep = line.strip().split(' ')
        epoch = line_sep[-1]
        eval_epoch_list.append(epoch)
      if 'Top1:' in line:
        line_sep = line.strip().split(' ')
        acc = line_sep[-1].split('%')[0]
        eval_top1_list.append(acc)
      if 'Top5:' in line:
        line_sep = line.strip().split(' ')
        acc = line_sep[-1].split('%')[0]   
        eval_top5_list.append(acc)     
        # print(epoch, acc)     
    for i in eval_epoch_list:
        print(i)
    for i in eval_top1_list:
        print(i)