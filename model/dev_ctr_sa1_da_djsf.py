import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graph.ntu_rgb_d import Graph

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


class TemporalConv(nn.Module):
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
        self.num_joints = 25
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))  #Figure 4 D-JSF gama系数y

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

        # Additional Max & 1x1 branch
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
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        #全局通道上下文和局部通道上下文注意力特征融合，将原来的conv1x1换成了通道注意力
        # self.aff = AFF(out_channels)  #   上面已经完成了类似于ctrgcn中tcn的多尺度卷积，在这里再增加了注意力特征融合，增加了global,local特征的融合
        self.transform = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1))
        # initialize
        self.bn = nn.BatchNorm2d(out_channels)
        # self.drop = nn.Dropout(dropout, inplace=True)

        self.apply(weights_init)

    def forward(self, x):   # dgstgcn中的dsf模块
        # Input dim: (N,C,T,V)
        # res = self.residual(x)
        N, C, T, V = x.shape  # bs,C',T,,V [4, 64, 100, 25]
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)  # [4, 64, 100, 26] <- cat [4, 64, 100, 25] 在V上进行平均 [4, 64, 100, 1]

        branch_outs = []
        for tempconv in self.branches:  # 四分支 conv2d(64,16) conv2d(16,16), conv2d(64,16) conv2d(16,16), conv2d(64,16) conv2d(16,16), conv2d(64,16)
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)  # [4, 64, 64, 25] 已经完成concat了，再进行AFF即通道注意力
        local_feat = out[..., :V]  # [4, 64, 100, 25]  类似于TCA-GCN中的AFF模块，分别得到global,local特征
        global_feat = out[..., V]  # 只在V上即空间上进行了平均[4, 64, 100]  # TCA-GCN中的AFF模块是在T和V上进行平均池化了[4,64,1,1]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])  # D-JSF步骤 [4, 64, 100, 25] <- [4, 64, 100]<- gama系数self.add_coeff[:V] [25]都是0
        out = local_feat + global_feat    # [4, 64, 100, 25]
        out = self.transform(out)  # [4, 64, 100, 25] <- 全连接吧 conv2d(64,64) [4, 64, 100, 25]
        out = self.bn(out)
        # out = self.drop(out)
        # out += res
        # out = self.aff(out, res)  #self.residual(x) [4, 64, 64, 25] 进行残差连接
        return out

# sa1_aff中sa1是通过节点自相关性拓扑建模，sa1_da通过自相关性和之前的ctr联合起来进行建模
class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)  # (3,8)或者 (64,8)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)  # (3,8, k = [1,1])
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)  # conv2d(3,64, k = (1,1))
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)  # (8,64, k=(1,1))
        # self.conv5 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)  # (3,8)或者 (64,8)
        # self.da = nn.Parameter(torch.ones(1))
        # self.ctr = nn.Parameter(torch.ones(1))

        self.da = 1
        self.ctr = 1

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):  # x [bs*2, 3, step, 25] [4,3,64,25] A v,v [25,25]因为有三个ctrgc,所以将A分开了
        N,C,T,V = x.size()  # [4,3,64,25]
        # [4,3,64,25] -> [4,3,25] conv2d(3,8, k=(1,1)) -> [4,8,25]
        # x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x) # [4,8,25] <- conv2d(3,8) [4,3,25]
        x1 = self.conv1(x)  # x1 = [4,8,64,25] <- conv2d(3,8, k = (1,1)) [4,3,64,25]
        x2 = self.conv2(x)  # x1 = [4,8,64,25] <- conv2d(3,8) [4,3,64,25]
        x3 = self.conv3(x)  #  [4,64,64, 25]<- conv2d(3,64, k = (1,1)) [4,3,64,25]
        
        attention_da = self.softmax(torch.bmm(x1.view(-1, T, V).permute(0, 2, 1) , x2.view(-1, T, V)).view(N, -1, V, V)) #  [4,8,25,25]  bs,C,V,V <- [bsC,V,T] [bsC,T,V]
        attention_ctr = self.tanh(x1.mean(-2).unsqueeze(-1) - x2.mean(-2).unsqueeze(-2))  #[4,8,25,25]
        x1 = attention_da*self.da + attention_ctr*self.ctr # [4,64,25,25] <- conv2d(8,64) [4,8,25,25]
        # print('da: {}, ctr: {}'.format(self.da[0], self.ctr[0]))

        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V, x1是通道拓扑细化，A是静态拓扑 [4,64,25,25] <-conv2d(8,64) [4,8,25,25] 
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)  #   bs,C,T,V [4,64,64,25] <- bs,C,T,V[4,64,64,25]  bs,C,V,V[4,64,25,25] 通道细化后的节点自相关性
        return x1

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


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
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
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
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

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        #if graph is None:
        #    raise ValueError()
        #else:
        #    Graph = import_class(graph)
        #    self.graph = Graph(**graph_args)

        self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)  # [2*3*25]

        base_channel = 64

        # self.seginfo = SGN(self, num_classes, 64, bias=True)

        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape  # [bs, step, 25*3]

            # [bs, step, 25, 3] -> [bs, 3, step, 25] -> [bs, 3, step, 25, 1]
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()   # [bs, 3, step, 25, M(numperson:2)]

        # N,M,V,C,T -> N,MVC,T
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # [bs, 2*25*3, step]
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # [bs*2, 3, step, 25]
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)