from torchsummaryX import summary
from model.sectrgcn import Model
import torch
import yaml

cfg = {'model_args':{'num_class': 60, 'num_point': 25, 'num_person': 2, 'graph':'graph.ntu_rgb_d.Graph','graph_args':{'labeling_mode':'spatial'}}}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
ctrgcn=Model()

summary(ctrgcn,torch.randn(64, 3, 64, 25, 2))
