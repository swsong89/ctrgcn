import torch
from fps import measure_inference_speed
from model.sectrgcn import Model

net = Model()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
net = net.to(device)
bs = 1
print(bs)
data = torch.rand((bs, 3, 64, 25, 2)).to(device)
measure_inference_speed(net, (data,))
