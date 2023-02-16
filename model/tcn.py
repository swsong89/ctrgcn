import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from .init_func import bn_init, conv_init


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):  # [4, 14, 100, 26]已经进行 conv2d(64,14)了
        return self.drop(self.bn(self.conv(x)))  # [4, 14, 100, 26] <- Conv2d(14, 14, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)) [4, 14, 100, 26]

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass


class dgmstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg  # [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        num_branches = len(ms_cfg)  # 6 C/N,这里的N就是6，即最后分了10,10,10,10,10,146种 dynamic TCN
        self.num_branches = num_branches  # 6
        self.in_channels = in_channels  # 64
        self.out_channels = out_channels  # 64

        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))  #Figure 4 D-JSF gama系数y

        if mid_channels is None:
            mid_channels = out_channels // num_branches  # 10
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)  # 14
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels  # 64

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):  # bs,C',T,,V [4, 64, 100, 25]
        N, C, T, V = x.shape  # bs,C',T,,V [4, 64, 100, 25]
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)  # [4, 64, 100, 26] <- cat [4, 64, 100, 25] 在V上进行平均 [4, 64, 100, 1]

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)  # # [4, 14, 100, 26] <- [4, 64, 100, 26]
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)  # bs,C,T,V [4, 64, 100, 26] <- C分别为14(conv2d(64,14),conv2d(14,14),D=1), 10(conv2d(64,10),conv2d(10,10,),D=2),10(conv2d(64,10),conv2d(10,10),D=3),10(conv2d(64,10),D=4)),10（conv2d(64,10),maxpool2d), 10(conv2d(64,10))   6[4, C, 100, 26]
        local_feat = out[..., :V]  # [4, 64, 100, 25]  类似于TCA-GCN中的AFF模块，分别得到global,local特征
        global_feat = out[..., V]  # 只在V上即空间上进行了平均[4, 64, 100]  # TCA-GCN中的AFF模块是在T和V上进行平均池化了[4,64,1,1]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])  # D-JSF步骤 [4, 64, 100, 25] <- [4, 64, 100]<- gama系数self.add_coeff[:V] [25]都是0
        feat = local_feat + global_feat    # [4, 64, 100, 25]

        feat = self.transform(feat)  # [4, 64, 100, 25] <- 全连接吧 conv2d(64,64) [4, 64, 100, 25]
        return feat

    def forward(self, x):  #  [4, 64, 100, 25]
        out = self.inner_forward(x)  #  [4, 64, 100, 25]
        out = self.bn(out)
        return self.drop(out)  #  [4, 64, 100, 25]