import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, n_in, n_out, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.full = nn.Sequential(
            conv3x3(n_in, n_out, stride),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True),
            conv3x3(n_out, n_out),
            nn.BatchNorm2d(n_out)
        )
        self.skip = conv1x1(n_in, n_out)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out_ = self.full(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        identity = self.skip(identity)
        out_ += identity
        out = self.relu(out_)

        return out

def LinConv(n_in, n_out):
    return nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(n_out),
            nn.ReLU(True)
        )

class SimpleHourglass(nn.Module):
    def __init__(self, n_in, n_out, n):
        super(SimpleHourglass, self).__init__()
        
        self.upper_branch = nn.Sequential(
            ResBlock(n_in, 256),
            # ResBlock(256, 256),
            ResBlock(256, n_out)
        )
        
        self.lower_branch = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            ResBlock(n_in, 256),
            # ResBlock(256, 256),
            ResBlock(256, 256)
        )
        
        if n > 1:
            self.inter = SimpleHourglass(n_in, n_out, n-1)
        else:
            self.inter = ResBlock(256, n_out)
        self.last = ResBlock(n_out, n_out)
        self.laster = conv1x1(2*n_out, n_out)
        
    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.lower_branch(x)
        lower = self.inter(lower)
        lower = self.last(lower)
        lower = F.interpolate(lower, scale_factor=2)
        out = torch.cat((lower,upper),dim=1)
        out = self.laster(out)
        return out

class Hourglass(nn.Module):
    def __init__(self, n_in, n_out, n):
        super(Hourglass, self).__init__()
        self.upper_branch = nn.Sequential(
            ResBlock(n_in, 256),
            ResBlock(256, 256),
            ResBlock(256, n_out)
        )
        
        self.lower_branch = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            ResBlock(n_in, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        
        if n > 1:
            self.inter = Hourglass(256, n_out, n-1)
        else:
            self.inter = ResBlock(256, n_out)
        self.last = ResBlock(n_out, n_out)
        
    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.lower_branch(x)
        lower = self.inter(lower)
        lower = self.last(lower)
        lower = F.interpolate(lower, scale_factor=2)
        return 0.8*upper + 0.2*lower

def torchMaxCoords(hmap):
    idxs = hmap.view(*hmap.shape[:2], -1).max(dim=-1)[1]
    xs = idxs % 128
    ys = idxs // 128
    return torch.stack([ys,xs], dim=-1)

def torchPCK(hmap_pred, hmap_true, threshold=5):
    coords_true = torchMaxCoords(hmap_true)
    coords_pred = torchMaxCoords(hmap_pred)
    diff = coords_pred - coords_true
    dist = torch.sqrt((diff ** 2).sum(dim=-1).float())
    return (dist < threshold).view(-1).float().mean().item()


class SoftArgmax(nn.Module):
    def __init__(self, img_shape):
        super(SoftArgmax, self).__init__()
        dim = img_shape[0]
        X,Y = np.meshgrid(
            range(1, dim+1),
            range(1, dim+1)
        )
        X = torch.from_numpy(X / dim).type(torch.FloatTensor)
        Y = torch.from_numpy(Y / dim).type(torch.FloatTensor)
        
        self.Wx = torch.nn.Parameter(X.view(1,-1))
        self.Wy = torch.nn.Parameter(Y.view(1,-1))
        self.d = 1/dim
        
    def forward(self, x):
        sm = F.softmax(
            x.flatten(start_dim=2), dim=2
        )
        coord_x = torch.bmm(self.Wx.expand_as(sm),
                            sm.transpose(2, 1)) - self.d
        coord_y = torch.bmm(self.Wy.expand_as(sm),
                            sm.transpose(2, 1)) - self.d

        coords = torch.stack([
            coord_y.view(-1, 1),
            coord_x.view(-1, 1)],
            dim = -1)
        return coords