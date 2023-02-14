from torchsummaryX import summary
# from model.sectrgcn import Model
# from model.ctrgcn import Model
from model.dev_ctr_sa1 import Model

import torch
import yaml

cfg = {'model_args':{'num_class': 60, 'num_point': 25, 'num_person': 2, 'graph':'graph.ntu_rgb_d.Graph','graph_args':{'labeling_mode':'spatial'}}}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
ctrgcn=Model().to(device)

summary(ctrgcn,torch.randn(64, 3, 64, 25, 2).to(device))
"""
ctrgcn
Total params            1.427912M
Trainable params        1.427912M
Non-trainable params          0.0
Mult-Adds             877.434614M

dev_ctr_sa1

Total params            1.427912M
Trainable params        1.427912M
Non-trainable params          0.0
Mult-Adds             877.434614M


dev_ctr_sa1_aff
Total params             2.493384M
Trainable params         2.493384M
Non-trainable params           0.0
Mult-Adds             1.166322934G


sectrgcn

Total params             1.838284M
Trainable params         1.838284M
Non-trainable params           0.0
Mult-Adds             1.455380664G
"""
