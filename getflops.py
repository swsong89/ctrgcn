from torchsummaryX import summary
# from model.sectrgcn import Model
from model.dev_ctr_sa1_aff_ta import Model
# from model.tca import Model
# from model.dev_ctr_sa1_aff import Model
from thop import profile


import torch
import yaml

cfg = {'model_args':{'num_class': 60, 'num_point': 25, 'num_person': 2, 'graph':'graph.ntu_rgb_d.Graph','graph_args':{'labeling_mode':'spatial'}}}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
model=Model().to(device)
# print(model)


input = torch.randn(1, 3, 64, 25, 2).to(device)


# flops, params = profile(model, (input, ))
# print('flops: {}M params: {}M'.format(flops/(input.size()[0]*input.size()[-1]*1000000), params/1000000))

summary(model, input)
"""
summary bathcsize不影响
ctrgcn
Total params            1.427912M
Trainable params        1.427912M
Non-trainable params          0.0
Mult-Adds             877.434614M
flops: 893.00608M params: 1.427912M

sa1 没有增加什么参数量和计算量
dev_ctr_sa1
Total params            1.427912M
Trainable params        1.427912M
Non-trainable params          0.0
Mult-Adds             877.434614M
flops: 893.84968M params: 1.427912M



dev_ctr_aff 主要增加了很多参数量1.06M, 计算量也增加到了877+289=1166M

                            Totals
Total params             2.493384M
Trainable params         2.493384M
Non-trainable params           0.0
Mult-Adds             1.166322934G

ta  增加参数量2.45M, 减少计算量367M
                         Totals
Total params          3.884432M
Trainable params      3.884432M
Non-trainable params        0.0
Mult-Adds             510.9245M

ta1  conv3 减少了参数量0.427M, 增加计算量314M
                          Totals
Total params           1.000952M
Trainable params       1.000952M
Non-trainable params         0.0
Mult-Adds             563.46611M


dev_ctr_sa1_aff  aff 主要增加了很多参数量1.06M, 计算量也增加到了877+289=1166M
Total params             2.493384M
Trainable params         2.493384M
Non-trainable params           0.0
Mult-Adds             1.166322934G
flops: 1191.96104M params: 2.493384M

dev_ctr_sa1_aff_ta
                          Totals
Total params           4.949904M
Trainable params       4.949904M
Non-trainable params         0.0
Mult-Adds             799.81282M

dev_ctr_sa1_aff_ta1 参数量为+0.63M, 计算量-20M
                          Totals
Total params           2.066424M
Trainable params       2.066424M
Non-trainable params         0.0
Mult-Adds             852.35443M

dev_ctr_sa1_aff_ta2  参数量为+0.63M, 计算量-20M
                           Totals
Total params            1.998744M
Trainable params        1.998744M
Non-trainable params          0.0
Mult-Adds             850.943734M

tca
                          Totals
Total params           4.949904M
Trainable params       4.949904M
Non-trainable params         0.0
Mult-Adds             799.81282M
flops: 829.84874M params: 4.298064M


hdgcn
                           Totals
Total params            1.546776M
Trainable params        1.546776M
Non-trainable params          0.0
Mult-Adds             810.165334M
flops: 945.089152M params: 1.632568M




sectrgcn

Total params             1.838284M
Trainable params         1.838284M
Non-trainable params           0.0
Mult-Adds             1.455380664G
"""