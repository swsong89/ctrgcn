import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer

from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=False,
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass


class unit_aagcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, attention=True):
        super(unit_aagcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

        if self.attention:
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A):

        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)


class unit_sgn(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: N, C, T, V; A: N, T, V, V
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))


class dggcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels  # 3
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets  # 8 即k为8 dynamic group
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise  # subset_wise是False,如果是True的话PA才会+DA+CA,False只有PA

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio  # 0.25 或者 0.125
        mid_channels = int(ratio * out_channels) # 0.125*64 = 8
        self.mid_channels = mid_channels  # 8

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())  # [8, 25, 25]

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))  # Figure3 (b)DA系数
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))  # Figure3 (b)DA系数

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)  # 输入3,输出8*8
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):  # # # bs,C,T,V [4, 3, 100, 25]
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape  # bs,C,T,V [4, 3, 100, 25]

        res = self.down(x)  #  [4, 64, 100, 25] <- conv2d(3,64) bs,C,T,V [4, 3, 100, 25]
        A = self.A  # K,V,V [8,25,25]  # 类似于细化后的A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]  # [1, 8, 1, 1, 25, 25]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)  #  bs, k, C',T,V [4,8,8,100,25] <-reshape [4, 64, 100, 25] <- pre conv2d(3,64)[4, 3, 100, 25]
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x  # bs,C,T,V [4, 3, 100, 25]

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)  # Figure 3(b) T-pooling bs,C,T,V[4, 3, 1, 25]

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)  #  bs,k,c',T,V [4,8,8,1,25]<-reshape [4, 64, 1, 25]<-conv2d(3,64) [4, 3, 1, 25]
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)  #  bs,k,c',T,V [4,8,8,1,25]<-reshape [4, 64, 1, 25]<-conv2d(3,64) [4, 3, 1, 25]

        if self.ctr is not None:  # CA模块  #  利用相减实现通道建模
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)  # ctrgcn部分的逻辑  bs,k,C',T,V,V[4, 8, 8, 1, 25, 25] <- [4, 8, 8, 1, 25, 1] - [4, 8, 8, 1, 1, 25]
            ada_graph = getattr(self, self.ctr_act)(diff)  # [4, 8, 8, 1, 25, 25] <- self.ctr_act 'tanh' [4, 8, 8, 1, 25, 25]

            if self.subset_wise:  # False
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]  # [4, 8, 8, 1, 25, 25] * 0
            A = ada_graph + A  # 对A没影响，A没有加DA

        if self.ada is not None:  # DA模块 利用自注意力实现拓扑[V,C] [C,V] -> [V,V]
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]  # [4, 8, 1, 1, 25, 25] <- [4, 8, 1, 25, 25] <- [4,8,8,1,25]  [4,8,8,1,25]
            ada_graph = getattr(self, self.ada_act)(ada_graph)  #N,K,C(1),T(1),V,V [4, 8, 1, 1, 25, 25]  <- self.ada_act softmax

            if self.subset_wise:  # False
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)  # bs,k,c',T,V,V[4, 8, 1, 1, 25, 25]
            else:
                ada_graph = ada_graph * self.beta[0]  # [4, 8, 1, 1, 25, 25]  CA权重beta
            A = ada_graph + A  # 没影响 A没有加CA bs,k,c',T,V，V[4, 8, 8, 1, 25, 25] <- A [4, 8, 8, 1, 25, 25] +  [4, 8, 1, 1, 25, 25]

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6  # bs,k,c(8),T(1),V,V[4, 8, 8, 1, 25, 25] 
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)  # # bs,k,c‘,V,V[4, 8, 8, 25, 25] 
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()  # bs,k,c',T,V[4, 8, 8, 100, 25] <- bs,k,c',T,V[[4, 8, 8, 100, 25]  bs,k,c‘,V,V[4, 8, 8, 25, 25] 
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        x = x.reshape(n, -1, t, v)  # bs,C',T,V[4, 64, 100, 25] <-reshape bs,k,C',T,V [4, 8, 8, 100, 25] Figure.3 DG-GCm output
        x = self.post(x)  #[4, 64, 100, 25] <- concat + 1x1conv步骤conv2d(64,64) [4, 64, 100, 25]
        return self.act(self.bn(x) + res)  # bs,C',T,,V [4, 64, 100, 25]残差连接，relu激活