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

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)  # Figure 3a ctrgcn中的concat步骤,即将5x1conv d=1, 5x1conv d=2, 3x1 maxpool拼接操作
        out += res  # 
        return out

class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in,out_channels, ratio, kernels, bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool2d((None,1))  # initial weights操作
        self.globalpool = nn.AdaptiveAvgPool2d(1)  # x到G(')操作
        self.g = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )  # conv2d(3,3, kernel_size=(1, 1),)  [4, 3, 1, 1]， 学习通道之间的关系
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in//ratio),
            kernel_size=[kernels[0],1],
            padding=[kernels[0]//2,0],
        )  # Conv2d(3, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)) [4, 3, 64, 1] -> bs,C,T,1 [4, 1, 64, 1] 将通道信息进行压缩，只剩下bs,T信息
        self.bn = nn.BatchNorm2d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)  # bs,C,T,1 [4, 1, 64, 1] 对C维度进行标准化，相当于是对bs,T进行标准化,增加数据的稳定性
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in//ratio),
            out_channels=out_channels,
            kernel_size=[kernels[1],1],
            padding=[kernels[1]//2,0],
            bias=False
        )  # Conv2d(1, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)，作用是对bs,C,T,1 [4, 1, 64, 1]相当于对T维度的权重学习64个通道-> [4,64,64,1]
        self.b.skip_init=True
        self.b.weight.data.zero_() 
        
    def forward(self, x):  # bs,C,T,V[4, 3, 64, 25] 
       # print('rf',x.shape)
        g = self.globalpool(x)  # C'Cx1x1 bs,C,1,1[4, 3, 1, 1] 在T,V维度进行平均池化,剩下的是通道的关系
       # print('rf1',g.shape)
        x = self.avgpool(x)  # bs,C,T,1 [4, 3, 64, 1] 在V维度进行平均池化， 剩下的是时间帧通道的关系
       # print('rf2',x.shape,(x+self.g(g)).shape)  下面是利用上下文信息校准时间维度上的权重，进行时间维度上特征聚合
        x = self.a(x + self.g(g))  #  bs,C,T,1 [4, 1, 64, 1] <- 在T维度卷积 Conv2d(3, 1, kernel_size=(3, 1), ) [4,3,64,1] <- ([4, 3, 64, 1] + conv2d(3,3) [4, 3, 1, 1])
        x = self.bn(x)  # 
        x = self.relu(x)  # [4, 1, 64, 1]
        x = self.b(x) + 1  # [4, 64, 64, 1] <- 增加维度,以及学习前后帧权重关系，conv2d(1, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0) [4, 1, 64, 1]
        return x  # # bs,C‘,T,1 [4, 64, 64, 1] 得到的是G(') ,即学了64个通道，每个通道都有T的相互关系权重，1是类似于MPS增加前后帧的相关性，因为MPS说了自注意力会关注距离比较远的帧，所以需要+1来平衡


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
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)  # 将维减少计算量
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)  # # conv2d(3,64, k = (1,1))
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)  # 将维减少计算量

        self.tanh = nn.Tanh()
        
        self.conv_rf = RouteFuncMLP(c_in= self.rel_channels,  out_channels= self.rel_channels,          # number of input filters
                    ratio=1,            # reduction ratio for MLP
                    kernels=[3,3],      # list of temporal kernel sizes
        )  # bs,C‘,T,1 [4, 64, 64, 1] 论文中G(')生成T维度之间的权重
        # self.conv = TAdaAggregation(
        #             in_channels     =in_channels,
        #             out_channels    =out_channels,
        #             kernel_size     = 1, 
        #             stride          = 1, 
        #             padding         = 0, 
        #             bias            = False,
                    
        #         )  # Temporal Aggregation   输入x和self.conv_rf(x) bs,C',T,1 [4, 64, 64, 1]
        # #self.gc1 = Graphsn_GCN(in_channels, out_channels)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):  # bs,C,T,V[4, 3, 64, 25] [25, 25]  channel-wise
        x1, x2 = self.conv1(x).mean(-2), self.conv2(x).mean(-2)  # bs,C,V[4, 8, 25]  [4, 8, 25]


        x3 = self.conv5(x)  # [4,8,64,25]
        # a_x3 = self.conv_rf(x)  # [4,8,64,1]
        x3 = x3 * self.conv_rf(x3)  # [4,8,64,25] <-[4,8,64,25] * [4,8,64,1]
        x3 = self.conv3(x3)  # bs,C,T,V [4,64,64,25] <- conv2d(64, 8) [4,8,64,25]



        # x3= self.conv(x, self.conv_rf(x))  #  #  类似于ctrgc中的conv1x1时间维度, bs,c',T,V [4, 64, 64, 25]  <-时间聚合TAdaAggregation (下面x特征[4, 3, 64, 25] ,上边W(') bs,C',T,1 [4, 64, 64, 1] )
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))  # K(')  # 学习了自相关性,当前节点与其他节点的距离，即边,即类似于A,所以叫通道拓扑，之前的A是静态拓扑[4, 8, 25, 25] <- conv2d(8,64)  ([4, 8, 25, 1] -[4,8,1,25] )
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0) #  refine A操作 [4, 64, 25, 25] <- [4, 8, 25, 25]  +  [1,1,25,25]
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)  # channel aggregation bs,C',T,V [4, 64, 64, 25] <- bs,C',V,V[4, 64, 25, 25]  bs,C',T,V [4, 64, 64, 25]
        return x1  # bs,C',T,V [4, 64, 64, 25]

class unit_tcn(nn.Module):  # 只是起到了一个残差连接的作用，是Figure 3 (a) add+的右边项
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