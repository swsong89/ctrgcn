import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import einsum
from einops import rearrange, repeat
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

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


# class CTRGC(nn.Module):
#     def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
#         super(CTRGC, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if in_channels == 3 or in_channels == 9:
#             self.rel_channels = 8
#             self.mid_channels = 16
#         else:
#             self.rel_channels = in_channels // rel_reduction
#             self.mid_channels = in_channels // mid_reduction
#         self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
#         self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
#         self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#         self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
#         self.tanh = nn.Tanh()
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 conv_init(m)
#             elif isinstance(m, nn.BatchNorm2d):
#                 bn_init(m, 1)

#     def forward(self, x, A=None, alpha=1):
#         x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
#         x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
#         x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
#         x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
#         return x1




# class unit_gcn(nn.Module):
#     def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
#         super(unit_gcn, self).__init__()
#         inter_channels = out_channels // coff_embedding
#         self.inter_c = inter_channels
#         self.out_c = out_channels
#         self.in_c = in_channels
#         self.adaptive = adaptive
#         self.num_subset = A.shape[0]
#         self.convs = nn.ModuleList()
#         for i in range(self.num_subset):
#             self.convs.append(CTRGC(in_channels, out_channels))

#         if residual:
#             if in_channels != out_channels:
#                 self.down = nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, 1),
#                     nn.BatchNorm2d(out_channels)
#                 )
#             else:
#                 self.down = lambda x: x
#         else:
#             self.down = lambda x: 0
#         if self.adaptive:  # 自适应性的话A就会更新
#             self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
#         else:
#             self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
#         self.alpha = nn.Parameter(torch.zeros(1))
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.soft = nn.Softmax(-2)
#         self.relu = nn.ReLU(inplace=True)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 conv_init(m)
#             elif isinstance(m, nn.BatchNorm2d):
#                 bn_init(m, 1)
#         bn_init(self.bn, 1e-6)

#     def forward(self, x):
#         y = None
#         if self.adaptive:
#             A = self.PA
#         else:
#             A = self.A.cuda(x.get_device())
#         for i in range(self.num_subset):
#             z = self.convs[i](x, A[i], self.alpha)
#             y = z + y if y is not None else z
#         y = self.bn(y)
#         y += self.down(x)
#         y = self.relu(y)


#         return y

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):  #  64, 8 , 3
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5  # 0.3535 = 8 ** -0.5
        inner_dim = hidden_dim * n_heads  # 24 = 8 * 3
        self.to_qk = nn.Linear(in_channels, inner_dim*2)  # Linear(in_features=64, out_features=48, bias=True)
        self.n_heads = n_heads  #  3
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):  # x bs,C,T,V[4, 64, 64, 25]
        y = rearrange(x, 'n c t v -> n t v c').contiguous()  # n,T,v, C [4, 64, 25, 64]
        y = self.ln(y)
        y = self.to_qk(y)  # [4, 64, 25, 48] <- conv2d(64,48) [4, 64, 25, 64]
        qk = y.chunk(2, dim=-1)  # [[4, 64, 25, 24], bs,T,V,C(hd)[4, 64, 25, 24]]
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)  # bs*T,h,v,d[256, 3, 25, 8]

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale  # [256, 3, 25, 25] <- [256, 3, 25, 8] [256, 3, 25, 8]
        attn = dots.softmax(dim=-1).float()  # bs*T,h,V,V[256, 3, 25, 25]  最后一个维度和为1
        return attn

class SA_GC(nn.Module):  # 相当于 unit_gcn
    def __init__(self, in_channels, out_channels, A, adaptive=True):  #  # 64, 64,A
        super(SA_GC, self).__init__()
        self.out_c = out_channels  # 64
        self.in_c = in_channels  # 64
        self.num_head= A.shape[0]  # 3
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)  #  [3,25,25] 对角线矩阵

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):  # 类似于三个ctrgc
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))  # Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8  # 8 = 64 // 8
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)  # 64, 8 , 3


    def forward(self, x, attn=None):  # bs,C,T,V [4, 64, 64, 25]
        N, C, T, V = x.size()

        out = None
        if attn is None:  # self.shared_topology[3,25,25],节点本身，入度，出度
            attn = self.attn(x)  # Figure2 右边SA(Ht)步骤 V->V softmax,权重 bs*T,h,V,V[256, 3, 25, 25]  一行和是1,该节点对其余节点的权重，出度<- conv2d(64, 48)  [4, 64, 64, 25]
        A = attn * self.shared_topology.unsqueeze(0)  # context-dependent Topology步骤 作用是对静态拓扑结构学习了一个自注意力权重，即该节点对所有节点的权重，但是相乘后即只对相连的edge其作用，bs*T,V,V[256, 3, 25, 25] <- # bs*T,hidden,V,V[256, 3, 25, 25] [1, 3, 25, 25]
        for h in range(self.num_head):    # 3
            A_h = A[:, h, :, :] # (nt)vv  [256, 25, 25]
            feature = rearrange(x, 'n c t v -> (n t) v c')  #[256, 25, 64] <- [4, 64, 64, 25]
            z = A_h@feature  # Selft-Attention graph conv  @矩阵相乘等于A_h.matmul(feature)     bs*T,V,C[256, 25, 64] <-  bs*T,V,V[256, 25, 25]  bs*V,C[256, 25, 64]
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()  # [4, 64, 64, 25]
            z = self.conv_d[h](z)  #[4, 64, 64, 25]  <-[4, 64, 64, 25] Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out  # [4, 64, 64, 25]


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

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = SA_GC(in_channels, out_channels, A, adaptive=adaptive)
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

        # A = self.graph.A # 3,25,25
        num_head = 3
        k = 1
        A = np.stack([np.eye(num_point)] * num_head, axis=0)  # A(3, 25, 25) num_point 25  num_head3  三个都是对角线

        print('adaptive: ', adaptive)

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)  # [2*3*25]

        base_channel = 64
        self.A_vector = self.get_A(graph, k)  #  [25, 25] 对角线减入度  
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)  # Linear(in_features=3, out_features=64, bias=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))  # [1, 25, 64]



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

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary  # A [25, 25] 从外向根节点 [ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  20  根节点是21(20+1)
        I = np.eye(Graph.num_node) # 25         #               [ 1, 20,  2, 20,  4,  5,  6, 20,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18,  1, 22,  7, 24, 11]  1
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))  # 对角线减入度


    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape  # [bs, step, 25*3]

            # [bs, step, 25, 3] -> [bs, 3, step, 25] -> [bs, 3, step, 25, 1]
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()   # [bs, 3, step, 25, M(numperson:2)]

        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()  # [256, 25, 3]
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x  # [256, 25, 3] <- [256, 25, 25] @ X [256, 25, 3] <-  256[25,25] @ X  # *是Hadamard   @ 矩阵相乘  self.A_vector 对角线减入度

        x = self.to_joint_embedding(x)  # [256, 25, 64] <-  [256, 25, 3] conv2d(3,64)
        x += self.pos_embedding[:, :self.num_point]  # [256, 25, 64] <- [256, 25, 64] + [1, :25, 64]
        
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