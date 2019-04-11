import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1:
			raise ValueError('BasicBlock only supports groups=1')
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
		super().__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = norm_layer(planes)
		self.conv2 = conv3x3(planes, planes, stride, groups)
		self.bn2 = norm_layer(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

def LinConv(n_in, n_out):
	return nn.Sequential(
			nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(n_out),
			nn.ReLU(True)
		)


class Hourglass(nn.Module):
	def __init__(self, n_in, n_out, n):
		super().__init__()
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

def coordPCK(threshold=5):
	def f(coords_pred, coords_true):
		diff = coords_pred - coords_true
		dist = torch.sqrt((diff ** 2).sum(dim=-1).float())
		return (dist < threshold).view(-1).float().mean().item()
	return f


class SoftArgmax(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        dim = img_shape
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
        coord_x = (self.Wx.expand_as(sm) * sm).sum(dim=2) - self.d
        coord_y = (self.Wy.expand_as(sm) * sm).sum(dim=2) - self.d

        coords = torch.stack([
            coord_y,
            coord_x],
            dim = -1)
        return coords
