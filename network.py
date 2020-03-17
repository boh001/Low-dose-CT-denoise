import numpy as np
import torch.nn as nn
import torch

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x.clone()
        out = self.relu(self.conv_first(x))
        out = self.relu(self.conv(out))
        residual_2 = out.clone()
        out = self.relu(self.conv(out))
        out = self.relu(self.conv(out))
        residual_3 = out.clone()
        out = self.relu(self.conv(out))

        # decoder
        out = self.conv_t(out)
        out += residual_3
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_2
        out = self.conv_t(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out



class W_Generator(nn.Module):
    def __init__(self, out_ch=32):
        super(W_Generator, self).__init__()

        self.main = [nn.Conv2d(1, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, 1, 3, 1, padding=1),
                nn.ReLU()]
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        out = self.main(x)
        return out

#inplace 해줘야하나
class W_Discriminator(nn.Module):
    def __init__(self):
        super(W_Discriminator, self).__init__()

        self.main = [nn.Conv2d(1, 64, 3, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 3, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 3, 2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 3, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, 3, 2),
                nn.LeakyReLU(0.2)
                ]
        self.main2 = [nn.Linear(12544, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1)
                ]
        self.main = nn.Sequential(*self.main)
        self.main2 = nn.Sequential(*self.main2)

    def forward(self, x):
        out = self.main(x)
        out = out.view(-1, 12544)
        out = self.main2(out)

        return out