import torch
import torch.nn.functional as F

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


input = torch.randn(3,4)
print(input)
b = F.softmax(input,dim=-1) # 按列SoftMax,列和为1
c = F.softmax(input) # 按列SoftMax,列和为1
print(b)
print(c)

