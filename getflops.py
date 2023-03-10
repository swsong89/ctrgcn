from torchsummaryX import summary
# from model.sectrgcn import Model
from model.dev_ctr_sa1_da_aff import Model
# from model.tca import Model
# from model.dev_ctr_sa1_aff import Model
# from thop import profile


import torch
import yaml

cfg = {'model_args':{'num_class': 120, 'num_point': 25, 'num_person': 2, 'graph':'graph.ntu_rgb_d.Graph','graph_args':{'labeling_mode':'spatial'}}}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
model=Model().to(device)
# model=Model(**cfg).to(device) # **cfg才会传参

# print(model)


input = torch.randn(1, 3, 64, 25, 2).to(device)

method = None

method = 'thop'

if method == 'thop':
  from thop import profile
  # thop
  flops, params = profile(model, (input, ))
  print('flops: {}M params: {}M'.format(flops/(input.size()[0]*input.size()[-1]*1000000), params/1000000))
else:
  # torchsummary
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


dev_ctr_dg
Total params             1.695688M
Trainable params         1.695688M
Non-trainable params           0.0
Mult-Adds             1.032167222G


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

dev_ctr_dg
                           Totals
Total params            1.622124M
Trainable params        1.622124M
Non-trainable params          0.0
Mult-Adds             993.472935M



dev_ctr_sa1_da_aff
                            Totals
Total params             2.493384M
Trainable params         2.493384M
Non-trainable params           0.0
Mult-Adds             1.166322934G








sectrgcn

Total params             1.838284M
Trainable params         1.838284M
Non-trainable params           0.0
Mult-Adds             1.455380664G


实验需要的
dev_ctr_sa1_da_aff
                            Totals
Total params             2.493384M
Trainable params         2.493384M
Non-trainable params           0.0
Mult-Adds             1.166322934G
flops: 1191.96104M params: 2.493384M

joint流, 上边的都是bone流


hdgcn论文中
          X-Sub (%) X-Set (%) GFLOPs # Param. (M)
DC-GCN [3]  84.0?   86.1?      1.83     3.37
MS-G3D [15] 84.9?   86.8?      5.22     3.22
CTR-GCN [1] 84.9    86.5?      1.97     1.46
InfoGCN [5] 85.1    86.3       1.84     1.57
HD-GCN      85.7    87.3       1.77     1.68




论文模型bone流


tca
                          Totals
Total params           4.949904M
Trainable params       4.949904M
Non-trainable params         0.0
Mult-Adds             799.81282M
flops: 829.84874M params: 4.298064M

ctrgcn
Total params            1.427912M
Trainable params        1.427912M
Non-trainable params          0.0
Mult-Adds             877.434614M
flops: 893.00608M params: 1.427912M

dgmsgcn
Total params             2.556366M
Trainable params         2.556366M
Non-trainable params           0.0
Mult-Adds             1.103712166G

hdgcn
                           Totals
Total params            1.546776M
Trainable params        1.546776M
Non-trainable params          0.0
Mult-Adds             810.165334M
flops: 945.089152M params: 1.632568M

infogcn
                           Totals
Total params            1.553752M
Trainable params        1.553752M
Non-trainable params          0.0
Mult-Adds             725.375792M
flops: 832.011264M params: 1.553752M


STGCN 论文说参数是2.4M
                            Totals
Total params             4.199242M
Trainable params         4.199242M
Non-trainable params           0.0
Mult-Adds             1.469820992G
"""