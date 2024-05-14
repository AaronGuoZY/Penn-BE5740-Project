import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DCB, UCB, Transformer

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # -------- ENCODER --------
        # conv layer
        self.encoder_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)   
        self.dcb_0 = DCB(in_channels = 32, out_channels=32)
        self.dcb_1 = DCB(in_channels = 32,out_channels=64)
        self.dcb_2 = DCB(in_channels = 64,out_channels=128)
        self.dcb_3 = DCB(in_channels = 128, out_channels=256)

        # -------- Transformer --------
        self.transformer_layer = Transformer()

        # -------- DECODER --------
        self.ucb0 = UCB(288, 128)
        self.ucb1 = UCB(128, 64)
        self.ucb2 = UCB(64, 32)
        self.ucb3 = UCB(32, 32)

        # -------- final conv layer --------
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, xi, h, a):
        # -------- ENCODER --------
        y0 = self.encoder_conv(xi)
        y0 = self.relu(y0)
        y1 = self.dcb_0(y0)
        y2 = self.dcb_1(y1)
        y3 = self.dcb_2(y2)
        y4 = self.dcb_3(y3)

        # -------- Transformer --------
        y5 = self.transformer_layer(y4, h, a)

        # -------- DECODER --------
        # before UCB, do an explicit concat
        w = torch.concat((y4, y5), dim=1)
        # UCB layers
        u0 = self.ucb0(w, y3)
        u1 = self.ucb1(u0, y2)
        u2 = self.ucb2(u1, y1)
        u3 = self.ucb3(u2, y0)

        # -------- final conv layer --------
        x_hat_o = self.final_conv(u3)
        x_hat_o = self.bn(x_hat_o)
        x_hat_o = x_hat_o + xi
        return x_hat_o


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # -------- ENCODER --------
        # conv layer
        self.encoder_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)   
        self.dcb_0 = DCB(in_channels = 32, out_channels=32)
        self.dcb_1 = DCB(in_channels = 32,out_channels=64)
        self.dcb_2 = DCB(in_channels = 64,out_channels=128)
        self.dcb_3 = DCB(in_channels = 128, out_channels=256)

        # -------- Transformer --------
        self.transformer_layer = Transformer()

        # -------- DECODER --------
        self.judge_conv_0 = nn.Conv2d(in_channels=288, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.judge_conv_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.judge_conv_2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)

        # -------- global average layer --------
        self.globalavg = nn.AvgPool2d(kernel_size=(13,10))

    def forward(self, xi, h, a):
        # -------- ENCODER --------
        y0 = self.encoder_conv(xi)
        y0 = self.relu(y0)
        y1 = self.dcb_0(y0)
        y2 = self.dcb_1(y1)
        y3 = self.dcb_2(y2)
        y4 = self.dcb_3(y3)

        # -------- Transformer --------
        y5 = self.transformer_layer(y4, h, a)

        # -------- Judge --------
        # first do an explicit concat
        w = torch.concat((y4, y5), dim=1)
        u0 = self.judge_conv_0(w)
        u0 = self.relu(u0)
        u1 = self.judge_conv_1(u0)
        u1 = self.relu(u1)
        u2 = self.judge_conv_1(u1)
        u2 = self.relu(u2)
        u3 = self.judge_conv_2(u2)
        u3 = self.relu(u3)
        # print("u3:", u3.shape)
        output = self.globalavg(u3)

        return output
