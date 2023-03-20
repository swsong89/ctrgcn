# import torch
# import torch.nn.functional as F
# import torch.nn as nn
import numpy as np
# y = torch.arange(25).unsqueeze(-1)  # torch.range(0, spa)
# print(y)
# print(y.size())

# y_onehot = torch.FloatTensor(25, 25)  # [25, 25]
# print(y_onehot.size())

# y_onehot.zero_()  # 置为0
# print(y_onehot.size())

# y_onehot.scatter_(1, y, 1)
# print(y_onehot.size())

# y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)  # [1, 1, 25, 25]
# print(y_onehot.size())

# y_onehot = y_onehot.repeat(128, 64, 1, 1)  # 在某一纬度进行重复 [bs, tem, 25, 25]
# print(y_onehot.size())


# input = torch.randn(3,4)
# print(input)
# b = F.softmax(input,dim=-1) # 按列SoftMax,列和为1
# c = F.softmax(input) # 按列SoftMax,列和为1
# print(b)
# print(c)
x = 1
gamm_list = [0, 1, 2, 3, 5]
gamma = 5
epsion = 0.9
x_list = [0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
x = np.array(x_list)
x1 = 1-x

for gamma in gamm_list:
  q_x = epsion
  m_x = (1-x)**gamma
  log_x = np.log(x)
  y = -q_x*m_x*log_x

  q_x1 = (1-epsion)/60
  m_x1 = x1**gamma
  log_x1 = np.log(x1)
  y1 = -q_x1*m_x1*log_x1
  y = y + y1
  print('gamma: ', gamma)
  for i in y:
    print(i)
