from functools import lru_cache

from imageio import imread

import numpy as np
import cv2

import torch
import h5py

class CachedImageReader():
	@staticmethod
	@lru_cache(maxsize=6)
	def _read(path):
		return imread(path)
		
	def __init__(self, keys=['img','mask','depth']):
		self.keys = keys
		
	def __call__(self, sample):
		for k in self.keys:
			sample[k] = self._read(sample[k])
		return sample

def DepthDecoder():
	""" Converts a RGB-coded depth into float valued depth. """
	def f(sample):
		encoded = sample['depth']
		top_bits, bottom_bits = encoded[:,:,0], encoded[:,:,1]
		depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
		depth_map /= float(2**16 - 1)
		depth_map *= 5.0
		return depth_map
	return f

class NormalizeMeanStd(object):
	def __init__(self, hmap=True):
		self.hmap = hmap

	def __call__(self, sample):
		mean = np.r_[[0.485, 0.456, 0.406]]
		std  = np.r_[[0.229, 0.224, 0.225]]
		sample['img'] = (sample['img'].astype('float32') / 255 - mean) / std
		if self.hmap:
			sample['hmap'] = sample['hmap'].astype('float32') / sample['hmap'].max()
		return sample

class NormalizeMax(object):
	def __init__(self, keys=['img','hmap']):
		self.keys = keys

	def __call__(self, sample):
		for k in self.keys:
			maxval = sample[k].max()
			if maxval:
				sample[k] = sample[k].astype('float32') / maxval
		return sample
	
class Coords2Hmap():
	def __init__(self, sigma, shape=(128,128), coords_scaling=1):
		self.sigma = sigma
		self.hmap_shape = shape
		self.c_scale = coords_scaling

	def __call__(self, sample):
		hmap = np.zeros((*self.hmap_shape, 21),'float32')
		coords = sample['coords']
		
		hmap[np.clip((coords[:,0] * self.c_scale).astype('uint'), 0, self.hmap_shape[0]-1),
			 np.clip((coords[:,1] * self.c_scale).astype('uint'),0, self.hmap_shape[1]-1),
			 np.arange(21)] = 10
		
		sample['hmap'] = cv2.GaussianBlur(hmap, (35, 35), self.sigma)
		return sample

class CenterNCrop():
	def __init__(self, in_shape, out_size, pad_radius=30):
		self.in_shape = in_shape
		self.out_size = out_size
		self.pad_radius = pad_radius
		
	@staticmethod
	def _getCircle(coords):
		min_ = coords.min(axis=0)
		max_ = coords.max(axis=0)
		center = min_ + (max_ - min_) / 2
		radius = np.sqrt(((max_ - center)**2).sum())
		return center, radius
	
	@staticmethod
	def circle2BB(circle, pad_radius):
		cnt, rad = circle
		rad = rad + pad_radius
		ymin, ymax = int(cnt[0]-rad), int(cnt[0]+rad)
		xmin, xmax = int(cnt[1]-rad), int(cnt[1]+rad)
		return xmin, xmax, ymin, ymax
	
	def __call__(self, sample):
		"""
		Input {'img': (*in_shape,3), 'coords': (21,2), *}
		Output {'img': (out_size,out_size,3), 'coords': (21,2), *}
		"""
		img, coords = sample['img'], sample['coords']
		crcl = self._getCircle(coords)
		xmin, xmax, ymin, ymax = self.circle2BB(crcl, self.pad_radius)
		
		pmin, pmax = 0, 0
		if xmin < 0 or ymin < 0:
			pmin = np.abs(min(xmin, ymin))
		
		if xmax > self.in_shape[0] or ymax > self.in_shape[1]:
			pmax = max(xmax - self.in_shape[0], ymax - self.in_shape[1])

		sample['yx_min_max'] = np.r_[[ymin, xmin, ymax, xmax]]
		
		img_pad = np.pad(img, ((pmin, pmax), (pmin, pmax), (0,0)), mode='wrap')
		
		if 'mask' in sample:
			mask = sample['mask']
			mask_pad = np.pad(mask, ((pmin, pmax), (pmin, pmax)), mode='wrap')

		xmin += pmin
		ymin += pmin
		xmax += pmin
		ymax += pmin
		
		img_crop = img_pad[ymin:ymax,xmin:xmax,:]
		if 'mask' in sample:
			mask_crop = mask_pad[ymin:ymax, xmin:xmax]
		
		coords += np.c_[pmin, pmin].astype('uint')
		rescale = self.out_size / (xmax - xmin)
		img_resized = cv2.resize(img_crop, (self.out_size, self.out_size))
		if 'mask' in sample:
			mask_resized = cv2.resize(mask_crop, (self.out_size, self.out_size))
		coords = coords - np.c_[ymin, xmin]
		coords = coords*rescale
		
		sample['img'] = img_resized
		sample['coords'] = coords.round().astype('uint8')
		if 'mask' in sample:
			sample['mask'] = mask_resized
		return sample

class RandomCropMask():
	"""
	Makes a random crop of the image and segmentation mask so that the mask is not empty
	"""
	def __init__(self, in_shape, out_size_min, out_size_max):
		self.bounds = (out_size_min, out_size_max)
		self.in_shape = in_shape
	
	@staticmethod
	def _random_crop_idx(mask, rad, in_shape):
		idx = np.argwhere(mask)
		if len(idx)>0:
			choise = np.random.randint(0,len(idx))
			point = idx[choise]
		else:
			point = np.random.randint(rad, min(in_shape), size=2)

		yyxx = np.r_[
		point[0] - rad,
		point[0] + rad,
		point[1] - rad,
		point[1] + rad
		]
		
		if yyxx[0] < 0: yyxx[:2] -= yyxx[0]
		if yyxx[1] > in_shape[0]: yyxx[:2] -= (yyxx[1] - in_shape[0])
		if yyxx[2] < 0: yyxx[2:] -= yyxx[2]
		if yyxx[3] > in_shape[1]: yyxx[2:] -= (yyxx[3] - in_shape[1])
		
		return yyxx
		
	def __call__(self, sample):
		size = np.random.randint(*self.bounds)
		
		tlbr = self._random_crop_idx(sample['mask'], size//2, self.in_shape)
		
		sample['mask'] = sample['mask'][tlbr[0]:tlbr[1],tlbr[2]:tlbr[3]]
		sample['img'] = sample['img'][tlbr[0]:tlbr[1],tlbr[2]:tlbr[3],:]
		
		return sample

class ToTensor(object):
	"""
	convert ndarrays in sample to Tensors
	"""
	def __init__(self, keys=['img','hmap','mask']):
		self._keys = keys

	def __call__(self, sample):
		for k in self._keys:
			x = sample[k].transpose((2,0,1))
			sample[k] = torch.from_numpy(x).type(torch.float32)
		return sample

def AddMaxCoords(sample):
	idxs = sample['hmap'].view(sample['hmap'].shape[0], -1).max(dim=-1)[1]
	xs = idxs % 128
	ys = idxs // 128
	coords = torch.stack([ys,xs], dim=-1)
	sample['coords'] = coords.type(torch.FloatTensor) / 128
	return sample

class ToNparray(object):
	"""
	convert ndarrays in sample to Tensors
	"""
	def __call__(self, sample):
		img, hmap, mask = sample['img'], sample['hmap'], sample['mask']

		return {
			'img': img.astype('float32'),
			'hmap': hmap.astype('float32'),
			'mask': mask.astype('float32')
		}

def Resize(out_shape, keys=['img','mask','depth']):
	def f(sample):
		for k in keys:
			sample[k] = cv2.resize(sample[k], out_shape)
		return sample
	return f
