import torch
import torch.nn as nn

class AFF(nn.Module):  # TCA-GCN模块
    '''
    Only one input branch
    '''

    def __init__(self, in_channels, r=1):
        super(AFF, self).__init__()
        inter_channels = in_channels//r
        channels=in_channels
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )  # conv2d(64, 64)  conv2d(64, 64)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )  # AdaptiveAvgPool2d(output_size=1)   conv2d(64, 64)  conv2d(64, 64)

        self.sigmoid = nn.Sigmoid()
        # init 使用方法就是在需要的地方加上 self.aff = AFF(out_channels)  #   上面已经完成了类似于ctrgcn中tcn的多尺度卷积，在这里再增加了注意力特征融合，增加了global,local特征的融合
        # forward         out = self.aff(out, self.residual(x))  #self.residual(x) [4, 64, 64, 25] 进行残差连接

    def forward(self, x, residual):  #   bs,C',T,V [4, 64, 64, 25]  Attentional Feature Fusion 输入是类似于三个ctrgc+残差连接的结果，输出就是特征聚合
        xa = x + residual
        xl = self.local_att(xa)  # [4, 64, 64, 25] <- conv2d(64,64), conv2d(64,64), [4, 64, 64, 25]
        xg = self.global_att(xa)  # 在T,V上进行了avgpool, bs,C',T,V,[4, 64, 1, 1] <-  conv2d(64,64) ,conv2d(64,64), [4, 64, 1, 1] <- AdaptiveAvgPool2d [4, 64, 64, 25]
        xlg = xl + xg  #   bs,C',T,V [4, 64, 64, 25]
        wei = self.sigmoid(xlg)  # Attentional Feature Fusion.pdf

        xo = 2 * x * wei + 2 * residual * (1 - wei)  # 2 * residual * (1 - wei)为空，没有进行残差连接 residual 0 
       
        return xo  # [4, 64, 64, 25]