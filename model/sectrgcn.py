import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.seginfo import DRJointSpa
from model.seginfo import embed
from model.ctrgc import SECTRGC
from model.ctrgc import CTRGC
from graph.ntu_rgb_d import Graph

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):  # CTR-GCN时间卷积部分
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch  3x1 MaxPool分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))  # 右边1x1conv

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

# 原来这里有CTRGC，然后把这个移出去到ctrgc.py
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):  # 三个ctr-gc叠加的部分，即Figure3.(a)上部分
    def __init__(self, in_channels, out_channels, A, isFirstLayer, coff_embedding=4, adaptive=True, bs=128, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]    # A.shape 3,25,25,所以是3个
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):    # 3个ctrgc换成sectrgc
            if isFirstLayer:
                self.convs.append(SECTRGC(in_channels, out_channels, bs=bs))  # 创新点SECTRGC,整个网络有10层ctrgc，只有第一层使用了SECTRGC,别的照旧
            else:
                self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)  # adaptive false A不进行更新, True进行更新
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):  # Figure3.(a)部分，先计算三个ctrgc，然后相加add,BN, 再加残差连接，再Relu,论文图是相加add,BN,Relu,再残差连接
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z  # [bs, 64, step, 25]
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):  # CTRGCN部分 Figure3.(a)
    def __init__(self, in_channels, out_channels, A, isFirstLayer=0, stride=1, pre_stride=1, residual=True, adaptive=True, bs=128, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.dim1 = 256
        self.num_classes = 60
        self.seg = 64  # seg 骨架序列数
        num_joint = 25  # NTU-RGB-D关节数为25

        self.tem = self.one_hot(bs, self.seg // pre_stride, num_joint)  # [bs, 25, 64, 64] pre_stride前4层1,再4层2,后2层4
        self.tem = self.tem.permute(0, 3, 2, 1).cuda()  # [bs, 64, 64, 25]

        # embed组成：正则化-》1x1卷积-》Relu激活-》1x1卷积-》Relu激活  创新点帧信息引导的多尺度时间卷积
        self.tem_embed = embed(self.seg // pre_stride, out_channels, norm=False)  # seg 骨架序列数  input:[bs, 64, 64, 25]

        self.gcn1 = unit_gcn(in_channels, out_channels, A, isFirstLayer, adaptive=adaptive, bs=bs)  # 所有都有残差连接
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)  # MultiScale_TemporalConv实际上就是ctrgcn 时间建模部分 temporal modeling
        self.relu = nn.ReLU(inplace=True)
        if not residual:  # ctrgcn只有第一层不需要，其余9层都需要, l1, 即figure.3(a)右边残差线
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):  # 2,3,4  6,7,  9,10
            self.residual = lambda x: x

        else:  # 5, 8
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # gcn1:[bs, 64, 64, 25], tem:[bs, 64, 64, 25]  [2, 64, 64, 25]
        y = self.tem_embed(self.tem)
        gcn = self.gcn1(x)
        y = y + gcn   # 原论文直接是gcn,加了一个y，帧信息引导
        z = self.relu(self.tcn1(y) + self.residual(x))
        return z

    def one_hot(self, bs, spa, tem):  # bs, self.seg, num_joint

        y = torch.arange(spa).unsqueeze(-1)  # torch.range(0, spa)
        y_onehot = torch.FloatTensor(spa, spa)  # [seg, seg]

        y_onehot.zero_()  # 置为0
        # scatter_(input, dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input
        # dim = 0 表示按行放置
        # 具体参考：https://www.freesion.com/article/91341387111/
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)  # 在某一纬度进行重复

        return y_onehot


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, batch_size=64, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        # if graph is None:
        #    raise ValueError()
        # else:
            # Graph = import_class(graph)
            # self.graph = Graph(**graph_args)

        self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)  # [2*3*25]

        base_channel = 64
        bs = batch_size*num_person
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, isFirstLayer=1, residual=False, adaptive=adaptive, bs=bs)  # 相对于CTRGCN把第一层改成了SECTRGC,其余不变
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, bs=bs)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, bs=bs)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, bs=bs)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive, bs=bs)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, pre_stride=2, adaptive=adaptive, bs=bs) # 创新点加上了密集连接
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, pre_stride=2, adaptive=adaptive, bs=bs)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, pre_stride=2, stride=2, adaptive=adaptive, bs=bs)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, pre_stride=4, adaptive=adaptive, bs=bs)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, pre_stride=4, adaptive=adaptive, bs=bs)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):  # SGN输入: [bs, 3, 25, step]  合法的输入类型 bs,C,T,V,M [64, 3, 64, 25, 2]
        if len(x.shape) == 3:  # 如果dims 3则表明是N, T, VC需要处理一下转成合适的格式
            N, T, VC = x.shape  # [bs, step, 25*3]

            # [bs, step, 25, 3] -> [bs, 3, step, 25] -> [bs, 3, step, 25, 1]
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()   # [bs, 3, step, 25, M(numperson:2)]   M(numperson:2)是什么意思？

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # [bs, 2*25*3, step]
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # [bs*2, 3, step, 25]

        x = self.l1(x)  # input x [128, 3, 64, 25]
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V  M可能为1
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)  # N,M,c_new,TV -> N,c_new   c_new应该等于base_channel*4
        x = self.drop_out(x)

        return self.fc(x)