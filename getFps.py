import torch
from fps import measure_inference_speed
from model.dev_ctr_sa1_aff import Model

net = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
bs = 64
print(bs)
# print(net)
data = torch.rand((bs, 3, 64, 25, 2)).to(device)
measure_inference_speed(net, (data,))
# bs 1
# dev_ctr_sa1_aff Overall fps: 38.4 img / s, times per image: 26.1 ms / img
# ctr Overall fps: 48.0 img / s, times per image: 20.8 ms / img  推理一张26.1

# bs 64
# dev_ctr_sa1_aff Overall fps: 8.1 img / s, times per image: 123.6 ms / img
# ctr Overall fps: 10.2 img / s, times per image: 98.1 ms / img  # 推理664张,98.1
# sectr Overall fps: 7.1 img / s, times per image: 140.4 ms / img
