from layers import DCB, UCB, Transformer
import torch
from torchsummary import summary
import torch.nn as nn

h = torch.zeros((1, 2))
a = torch.zeros((1, 100))

transformer = Transformer()
# summary(transformer, [(h, a)], ((13, 10, 256)))


# x_transformer = torch.rand(1, 256, 13, 10)  # Example input tensor
# h = torch.zeros(1,2)  # Example auxiliary input
# a = torch.zeros(1,100)    # Another example auxiliary input
# output = transformer(x_transformer, h, a)
# print(output)


dcb_0 = DCB(in_channels = 32, out_channels=32)
dcb_1 = DCB(in_channels = 32,out_channels=64)
dcb_2 = DCB(in_channels = 64,out_channels=128)
dcb_3 = DCB(in_channels = 128, out_channels=256)

xi = torch.zeros((1, 32, 208, 160))

y_1 = dcb_0(xi)
print("y_1:",y_1.shape)
y_2 = dcb_1(y_1)
print("y_2:",y_2.shape)
y_3 = dcb_2(y_2)
print("y_3:",y_3.shape)
y_4 = dcb_3(y_3)
print("y_4:",y_4.shape)

y_5 = transformer(y_4, h, a)
print("y_5:",y_5.shape)

print("---------------- UCB test ----------------")

w = torch.cat((y_4, y_5), axis = 1)
print("w:",w.shape)

ucb_0 = UCB(288, 128)
ucb_1 = UCB(128, 64)
ucb_2 = UCB(64, 32)
ucb_3 = UCB(32, 32)

u_0 = ucb_0(w, y_3)
print("u_:", u_0.shape)
u_1 = ucb_1(u_0, y_2)
print("u_1:", u_1.shape)

u_2 = ucb_2(u_1, y_1)
print("u_2:", u_2.shape)

u_3 = ucb_3(u_2, xi)
print("u_3:", u_3.shape)

# # define global average
# global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
# output_tensor = global_avg_pool(u_3)
# output = output_tensor.view(output_tensor.size(0), -1)
# print("output:", output)