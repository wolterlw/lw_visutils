import os
import pickle
import pathlib
import re

import numpy as np
import cv2
from skimage import io #TODO: delete 
from scipy.io import loadmat
import random

import torch
from torch.utils.data import Dataset

import h5py
from tqdm import tqdm


class Hand():
	"""
	Helper class used to visualize hand joints and generate gaussian heatmaps
	"""
	colormap = {
			'wrist': 'w',
			'thumb': '#ffc857',
			'index': '#e9724c',
			'middle': '#c5283d',
			'ring': '#00fddc',
			'pinky': '#255f85',
		}

	def __init__(self, joint_array, img_idx):
		self.img_idx = img_idx
		self.fully_visible = (joint_array > 0).all()
		self.array = np.clip(joint_array[:,:2],0,319)
		self.joints = {
			'wrist': self.array[:1],
			'thumb': self.array[1:5],
			'index': self.array[5:9],
			'middle': self.array[9:13],
			'ring': self.array[13:17],
			'pinky': self.array[17:21],
		}
	
	def draw(self, axis):
		for k in self.joints.keys():
			axis.plot(
				self.joints[k][:,0],
				self.joints[k][:,1],
				c=self.colormap[k])
			axis.plot([self.joints['wrist'][0,0],self.joints[k][-1,0]],
					 [self.joints['wrist'][0,1],self.joints[k][-1,1]],
					 c=self.colormap[k])
			
	def getHeatmap(self, img_shape):
		heatmap = np.zeros((*img_shape[:2], 21),'float')
		heatmap[self.array[:,1].astype('uint'),
				self.array[:,0].astype('uint'),
				np.arange(21)] = 10
		heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1)
		heatmap /= heatmap.max()
		return heatmap
		
	def getCircle(self):
		min_ = self.array.min(axis=0)
		max_ = self.array.max(axis=0)
		center = min_ + (max_ - min_) / 2
		radius = np.sqrt(((max_ - center)**2).sum())
		return center, radius

# =============================== DATASET WRAPPERS ==================================

class HDFDataset():
	def __init__(self, file_path, dset='train', transform=None):
		assert dset in {'train','val'}
		hdf_file = h5py.File(file_path, 'r')
		self.X = hdf_file['X_' + dset][()]
		self.y = hdf_file['y_' + dset][()]
		# self.masks = hdf_file['Mask_'+ dset][()]
		self.trns = transform
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		sample = {
			'img': self.X[idx],
			'coords': self.y[idx],
			# 'mask': self.masks[idx]
		}
		return self.trns(sample) if self.trns else sample

class RHDDataset(Dataset):
	"""
	Dataset class that loads joint coordinates on initialization and
	provides functionality to load images and construct training samples
	on the fly
	"""
	def __init__(self, root_dir, dset='training', transform=None):
		super(RHDDataset, self).__init__()
		self.root_dir = root_dir + dset + '/'
		with open(self.root_dir + 'anno_' + dset + '.pickle','rb') as f:
			coords = pickle.load(f)
		
		lr_slc = [slice(21),slice(21,None)]
		
		self.hands = [
			[Hand(v['uv_vis'][slc], k) for slc in lr_slc]\
			for k,v in coords.items()
		]
		self.hands = [h for pair in self.hands for h in pair if h.fully_visible]
		
		self.img_path = self.root_dir + dset + '/'
		self.transform = transform
	
	def __len__(self):
		return len(self.hands)
	
	def __getitem__(self, idx):
		hand = self.hands[idx]
		img_name = os.path.join(
			self.root_dir, 
			'color',
			str(hand.img_idx).zfill(5) + '.png')

		mask_name = os.path.join(
			self.root_dir,
			'mask',
			str(hand.img_idx).zfill(5) + '.png')
			
		img = cv2.imread(img_name)[:,:,::-1]
		mask = cv2.imread(mask_name,0)
		
		sample = {
			'img': img,
			'coords': hand.array[:,::-1],
			'mask': mask,
		}
	
		if self.transform:
			sample = self.transform(sample)
		return sample

class GANeratedDataset(Dataset):
	"""
	Dataset class that loads joint coordinates on initialization and
	provides functionality to load images and construct training samples
	on the fly
	"""
	_joint_reindex = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
	
	def __init__(self, root_dir, transform=None, preload_coords=False):
		super(GANeratedDataset, self).__init__()
		self.root_dir = pathlib.Path(root_dir).expanduser()
		
		self.joint_path = sorted([x.as_posix() for x in self.root_dir.glob('./*/*_joint2D.txt')])
		self.image_path = sorted([x.as_posix() for x in self.root_dir.glob('./*/*.png')])
		
		assert len(self.joint_path) == len(self.image_path), "joint-image mismatch"
		
		self.transform = transform
		
		if preload_coords:
			self.all_coords = [self._read_joints(x) for x in tqdm(self.joint_path)]
		
		
	@classmethod
	def _read_joints(cls, filename):
		with open(filename,'r') as f:
			content = f.readline()
		joint_arr = [float(x) for x in content.replace('\n','').split(',')]
		assert len(joint_arr) == 42, filename
		joints = np.r_[joint_arr].reshape(21,2)[:,::-1]
		return joints[cls._joint_reindex]
			
	def __len__(self):
		return len(self.joint_path)

	def __getitem__(self, idx):
		image = io.imread(self.image_path[idx])
		coords = self.all_coords[idx] \
			if self.all_coords else self._read_joints(self.joint_path[idx])
		
		sample = {
			'img': image,
			'coords': coords,
		}

		if self.transform:
			sample = self.transform(sample)
		return sample

class STDDatasetRAW(Dataset):
	def __init__(self, root_folder='~/my_data/STD', subset='B1Random', transforms=None):

		self.transforms = transforms
		self.K = np.r_[
			[[607.92271,0,314.78337]
			,[0,607.88192,236.42484]
			,[0,0,1]]
		]
		self.T = np.r_[[-24.0381, -0.4563, -1.2326]]
		
		rV = np.r_[[0.00531, -0.01196, 0.00301]]
		self.R = self._getR(rV)
		
		self.root_dir = pathlib.Path(root_folder).expanduser()
		self.img_files = sorted(
			self.root_dir.joinpath(subset).glob('./SK_color*.png'),
			key=lambda x: int(re.findall('[0-9]+', x.name)[0])
		)
		data_file = self.root_dir.joinpath('labels/' + subset + '_SK.mat')
		joint_data = loadmat(data_file.as_posix())
		self.joints3D = self._align3D(joint_data['handPara'].transpose(2,1,0))
		
		tmp_coords = np.arange(21)
		self.reidx_coords = np.concatenate(
			[tmp_coords[:1]]
			+ [tmp_coords[f+3:f-1:-1] for f in [17,13,9,5,1]]
		)
		
	def __len__(self):
		return len(self.joints3D)
				
	@staticmethod
	def _getR(rV):
		Wx = np.zeros((3,3))
		Wx[1,2] = -1
		Wx[2,1] = 1

		Wy = np.zeros((3,3))
		Wy[0,2] = 1
		Wy[2,0] = -1

		Wz = np.zeros((3,3))
		Wz[0,1] = -1
		Wz[1,0] = 1
		
		Rx = np.eye(3) + Wx * np.sin(rV[0]) + Wx**2 * (1-np.cos(rV[0]))
		Ry = np.eye(3) + Wy * np.sin(rV[1]) + Wy**2 * (1-np.cos(rV[1]))
		Rz = np.eye(3) + Wz * np.sin(rV[2]) + Wz**2 * (1-np.cos(rV[2]))
		return Rx @ Ry @ Rz
		
		
	def _align3D(self, points):
		rotated = (points @ self.R)
		translated = (rotated - self.T[np.newaxis, np.newaxis, :])
		return translated
	
	def _project(self, points):
		points = points @ self.K.T
		normalized = points[:,:2] / points[:,2:]
		assert np.all(normalized > 0), "negative coords"
		return normalized.round().astype('uint16')
	
	def __getitem__(self, idx):
		img = io.imread(self.img_files[idx])
		points = self._project(self.joints3D[idx])[:,::-1]
		points = points[self.reidx_coords,:]

		sample = {'img': img, 'coords': points}
		return self.transforms(sample) if self.transforms else sample

# ============================== PREPROCESSING METHODS ==============================

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
	def __init__(self, hmap=True):
		self.hmap = hmap

	def __call__(self, sample):
		sample['img'] = sample['img'].astype('float32') / 255
		if self.hmap:
			sample['hmap'] = sample['hmap'].astype('float32') / sample['hmap'].max()
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
	
# class Rotate():
# 	"""
# 	Rotate both image and heatmaps on a random angle from rotation_range,
# 	"""
# 	def __init__(self, rotation_range=(-25,25), keys=['img','hmap','mask']):
# 		self._keys = keys
# 		self.rotation_range = rotation_range
		
# 	def __call__(self, sample):
# 		angle = random.uniform(*self.rotation_range)
# 		for ent in self._keys:
# 			sample[ent] = np.abs(ndimage.rotate(sample[ent], angle, reshape=False))
# 		return sample

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