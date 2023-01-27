# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch import nn
import torch
import math

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
# device = torch.device("cpu")

# output:[bs, 128, 25, step]
class DRJointSpa(nn.Module):
    def __init__(self, bs=128, bias=True, device=0):
        super(DRJointSpa, self).__init__()

        self.num_classes = 60
        self.dim1 = 256
        num_joint = 25  # NTU-RGB-D关节数为25
        seg = 64
        self.spa = self.one_hot(bs, num_joint, seg)  # [bs, seg, num_joint, num_joint] [128, 64, 25, 25]
        # self.spa = self.spa.permute(0, 3, 2, 1).cuda()# [128, 25, 25, 64]  [bs, num_joint, num_joint, seg] 
        self.spa = self.spa.permute(0, 3, 2, 1).to(device)# [128, 25, 25, 64]  [bs, num_joint, num_joint, seg] 

        # embed组成：正则化-》1x1卷积-》Relu激活-》1x1卷积-》Relu激活
        # self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)  # seg 骨架序列数
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)

        # 卷积-》卷积-》softmax映射为0-1之间的实数，并且归一化保证和为1，因此多分类的概率之和也刚好为
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)

        self.gcn = gcn_spa(self.dim1 // 2, self.dim1 // 4, bias=bias)  # 图卷积 (128, 128)

    def forward(self, input):  # input: [bs*2, 3, step, 25]  [128, 3, 25, 64
        # print('input.shape: ', input.shape)  # [128, 3, 25, 64]
        bs, c, num_joints, step = input.size()
        # Dynamic Representation
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]  # 动力学信息， 前一帧-后一帧
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)  # dim=0, 横向连接，dim=1,纵向连接
        pos = self.joint_embed(input)  # 空间姿势信息  为什么经过一系列操作就变成了关节语义信息？？ [bs, 64, 25, step]
        spa1 = self.spa_embed(self.spa)  # 关节点语义信息 spa1:[bs, 128, 25, step]
        dif = self.dif_embed(dif)  # dif：动力学语义信息 [bs, 64, 25, step]
        dy = pos + dif  # 关节点信息+动力学信息 [bs, 64, 25, step]

        # Joint-level Module
        # print('dy: ', dy.shape)
        # print('spa1: ', spa1.shape)
        input = torch.cat([dy, spa1], 1)  # [bs, 128, 25, step]

        g = self.compute_g1(input)  # 相乘  input:[bs, 128, 25, step], g:[bs, step, 25, 25]
        input = self.gcn(input, g)
        return input  # [bs, 64, 25, step]

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)  # torch.range(0, spa)

        y_onehot = torch.FloatTensor(spa, spa)  # [25, 25]

        y_onehot.zero_()  # 置为0

        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)  # [1, 1, 25, 25]

        y_onehot = y_onehot.repeat(bs, tem, 1, 1)  # 在某一纬度进行重复 [bs, tem, 25, 25]

        return y_onehot


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * 25)  # dim*25 == 3*25 == c*num_joints

    def forward(self, x):  # x:[bs, 3, num_joints, step]
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)  # reshape
        x = self.bn(x)  # 批规范化
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),  # 3*25 批规范化
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):  # x:[bs, 3, num_joints, step]
        x = self.cnn(x)  # x:[bs, 128, 25, step]
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):  # (128, 64)
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):  # x1:[bs, 128, 25, step], g:[bs, step, 25, 25]
        x = x1.permute(0, 3, 2, 1).contiguous()  # [bs, step, 25, 128]
        x = g.matmul(x)  # x:[bs, step, 25, 128]
        x = x.permute(0, 3, 2, 1).contiguous()  # x:[bs, 128, 25, step]
        x = self.w(x) + self.w1(x1)  # x:[bs, 64, 25, step]
        # x = self.relu(self.bn(x))  # x:[bs, 64, 25, step]
        return x


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1  # 128
        self.dim2 = dim2  # 256
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):  # x1: [bs, 128, 25, step]

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()  # [bs, 256, 25, step] -> g1:[bs, step, 25, 256]
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()  # [bs, 256, 25, step] -> g2:[bs, step, 256, 25]
        g3 = g1.matmul(g2)  # bs和step提取出来，[25，256] * [256, 25] -> g3:[bs, step, 25, 25]
        g = self.softmax(g3)  # g:[bs, step, 25, 25]
        return g

