import torch
import torch.nn as nn

# UCB, DCB, 
# DCB block for encoder layer
class DCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DCB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)   
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        return x


class UCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, scale_factor=2, padding=1):
        super(UCB, self).__init__()
        # Use ConvTranspose2d for upsampling or a combination of Upsample and Conv2d
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        # self.upconv = nn.ConvTranspose2d(out_channels, out_channels/2, kernel_size=kernel_size,
        #                              stride=scale_factor, padding=1, output_padding=scale_factor - 1)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                            kernel_size=kernel_size, stride=scale_factor, padding=padding,output_padding=scale_factor - 1)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, d): # conv, conv, upconv
        # upconv, concatenate, conv, conv
        # print(x.shape)
        x = self.upconv(x)
        x = self.relu(x)
        # print(x.shape)
        x = torch.cat((x, d), dim=1)
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.shape)
        return x

# TRANSFORMER LAYERS TG
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm1d(130)
        self.bn1 = nn.InstanceNorm2d(32)
        self.bn2 = nn.InstanceNorm1d(130)
        self.flatten = nn.Flatten()  # Add a flatten layer
        # The output of bn2 will be concatenated with health state vector `h`
        self.dense2 = nn.Linear(132, 130)  # `+2` for health state vector
        # The output of fc2 will be concatenated with difference age vector `a_d`
        self.dense1 = nn.Linear(4160, 130)  # `+100` for difference age vector
        self.dense3 = nn.Linear(230, 4160)

    def forward(self, x, h, a):
        # print("initial h:", h.shape)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print("before batch norm", x.shape)
        x = self.bn2(x)
        # print("after batch norm", x.shape)
        # print("x:", x.shape)
        # print("h:", h.shape)
        x = torch.cat((x, h), dim = 1)
        # print(x.shape)
        x = self.dense2(x)
        # print("x",x.shape)
        # print("a",a.shape)
        x = torch.cat((x, a), dim = 1)
        # print("before dense 3", x.shape)
        x = self.dense3(x)
        # print(x.shape)
        x = torch.reshape(x, (-1, 32, 13, 10))
        # print(x.shape)
        return x