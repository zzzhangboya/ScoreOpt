import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_transpose_3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes,
                            kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True)

def conv3x3(in_planes, out_planes, stride=1):
    if stride < 0:
        return conv_transpose_3x3(in_planes, out_planes, stride=-stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_nonlin=True):
        super(BasicBlock, self).__init__()
        self.nonlin1 = Swish(planes)#nn.ELU()
        self.nonlin2 = Swish(planes)#nn.ELU()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.out_nonlin = out_nonlin

        self.shortcut_conv = None
        if stride != 1 or in_planes != self.expansion * planes:
            if stride < 0:
                self.shortcut_conv = nn.ConvTranspose2d(in_planes, self.expansion*planes,
                                                        kernel_size=1, stride=-stride,
                                                        output_padding=1, bias=True)
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes,
                                            kernel_size=1, stride=stride, bias=True)


    def forward(self, x):
        out = self.nonlin1(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut_conv is not None:
            out_sc = self.shortcut_conv(x)
            out += out_sc
        else:
            out += x
        if self.out_nonlin:
            out = self.nonlin2(out)
        return out

class MNISTResNet(nn.Module):
    def __init__(self, n_channels=64, quadratic=False):
        super().__init__()
        self.proj = nn.Conv2d(1, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(6)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.energy_linear = nn.Linear(n_channels, 1)
        self.energy_linear2 = nn.Linear(4 * 4 * n_channels, 1)
        self.energy_linear3 = nn.Linear(4 * 4 * n_channels, 1)
        self.quadratic = quadratic

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        input = self.proj(input)
        out = self.net(input)
        if self.quadratic:
            out = out.view(input.size(0), -1)
            return (self.energy_linear(out) * self.energy_linear2(out) + self.energy_linear3(out**2)).squeeze()
        else:
            out = out.view(out.size(0), out.size(1), -1).mean(-1)
            return self.energy_linear(out).squeeze()


class MNISTLeNet(nn.Module):
    def __init__(self):
        super(MNISTLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x

class MNISTConvNet(nn.Module):
    def __init__(self, nc=16, quadratic=False):
        super().__init__()
        self.quadratic = quadratic
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            nn.ELU(),

            nn.Conv2d(nc * 2, nc * 2, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            nn.ELU(),

            nn.Conv2d(nc * 4, nc * 4, 3, 1, 1),
            nn.ELU(),

            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            nn.ELU(),

            nn.Conv2d(nc * 8, nc * 8, 3, 1, 0),
            nn.ELU()
        )
        self.out = nn.Linear(nc * 8, 1)
        self.out2 = nn.Linear(nc * 8, 1)
        self.out3 = nn.Linear(nc * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        if self.quadratic:
            return (self.out(out) * self.out2(out) + self.out3(out**2)).squeeze()
        else:
            return self.out(out).squeeze()

class MNISTConvNetL(nn.Module):
    def __init__(self, nc=32, quadratic=False):
        super().__init__()
        self.quadratic = quadratic
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            nn.LeakyReLU(.2),

            nn.Conv2d(nc * 8, 1, 3, 1, 0),
        )

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        return out

class MNISTSmallConvNet(nn.Module):
    def __init__(self, nc=64, quadratic=False):
        super().__init__()
        self.quadratic = quadratic
        n_c = 1
        n_f = nc
        l = .2
        self.net = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            Swish(n_f),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            Swish(2 * n_f),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            Swish(4 * n_f),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            Swish(8 * n_f),
            nn.Conv2d(n_f * 8, n_f * 8, 3, 1, 0)
        )
        self.out = nn.Linear(n_f * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = self.out(out.squeeze()).squeeze()
        return out

class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)