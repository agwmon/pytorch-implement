import torch
import torch.nn as nn
import torch.nn.functional as F

# 첫번째 단에서 BN X, PReLU & LeakyReLU 있으므로 따로 module로 구현
class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, BN=True, act=None):
        super(conv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias))
        if BN:
            layers.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            layers.append(act)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels = 64, kernel_size = 3, act = nn.PReLU(64)):
        super(ResBlock, self).__init__()
        layers = []
        layers.append(conv(channels, channels, kernel_size, stride = 1, padding = 1, bias = True, BN = True, act = act))
        layers.append(conv(channels, channels, kernel_size, stride = 1, padding = 1, bias = True, BN = True, act = None))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.layers(x)
        x += res
        return x
    
class G_Body(nn.Module):
    def __init__(self, B):
        super(G_Body, self).__init__()
        layers = []
        self.conv1 = conv(3, 64, 9, 1, 4, bias = True, BN = False, act = nn.PReLU(64))
        for _ in range(B):
            layers.append(ResBlock(64, 3, act = nn.PReLU(64)))
        self.body = nn.Sequential(*layers)
        self.conv2 = conv(64, 64, 3, 1, 1, bias = True, BN = True, act = None)
    
    def forward(self, x):
        res = self.conv1(x)
        out = self.body(res)
        out = self.conv2(out)
        out += res
        return out

class upsample(nn.Module):
    def __init__(self, channels, upscaling):
        super(upsample, self).__init__()
        layers = []
        layers.append(conv(channels, channels * upscaling ** 2, 3, 1, bias = True, BN = False, act = None))
        layers.append(nn.PixelShuffle(upscaling))
        layers.append(nn.PReLU(channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class Generator(nn.Module):
    def __init__(self, num_blocks = 16, upscaling = 4):
        super(Generator, self).__init__()
        self.body = G_Body()
        upscaling_layers = [upsample(64, 2) for _ in range(2)]
        self.upscaling_layers = nn.Sequential(*upscaling_layers)
        self.conv = conv(64, 3, 9, 1, 4, bias = True, BN = False, act = nn.Tanh())

    def forward(self, x):
        x = self.body(x)
        x = self.upscaling_layers(x)
        x = self.conv(x)
        return x

class d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, act = nn.LeakyReLU(inplace = True)):
        super(d_block, self).__init__()
        layers = []
        layers.append(conv(in_channels, out_channels, kernel_size, 1, 1, bias = True, BN = True, act = act))
        layers.append(conv(out_channels, out_channels, kernel_size, 2, 1, bias = True, BN = True, act = act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

# 논문에서는 3 block => min patch_size = 16x16
class Discriminator(nn.Module):
    def __init__(self, c = 64, num_blocks = 3, patch_size = 64):
        super(Discriminator, self).__init__()
        self.act = nn.LeakyReLU(inplace = True)
        self.conv1 = conv(3, c, 3, 1, 1, True, False, self.act)
        self.conv2 = conv(64, c, 3, 2, 1, True, True, self.act)
        # body = [d_block(64, 128), d_block(128, 256), d_block(256, 512)]
        body = [d_block(c * (2 ** i), c * (2 ** (i+1)), 3, self.act) for i in range(num_blocks)]
        self.body = nn.Sequential(**body)
        # 64 // 2^4 = 4x4, 16 * (64 * 8)
        self.flat_size = ((patch_size // (2 ** (num_blocks + 1))) ** 2) * (c * (2 ** num_blocks))
        fc = [nn.Linear(patch_size, 1024), self.act, nn.Linear(1024, 1), nn.Sigmoid()]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.body(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
        
        