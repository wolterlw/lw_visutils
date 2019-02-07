import os
import pickle

import numpy as np
import cv2
from scipy import ndimage 
import random

import torch
from torch.utils.data import Dataset

import h5py


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
	
	def draw(self):
		for k in self.joints.keys():
			plt.plot(
				self.joints[k][:,0],
				self.joints[k][:,1],
				c=self.colormap[k])
			plt.plot([self.joints['wrist'][0,0],self.joints[k][-1,0]],
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

class HDFDataset():
    def __init__(self, file_path, dset='train', transform=None):
        assert dset in {'train','val'}
        hdf_file = h5py.File(file_path, 'r')
        self.X = hdf_file['X_' + dset][:1000]#[()]
        self.y = hdf_file['y_' + dset][:1000]#[()]
        self.masks = hdf_file['Mask_'+ dset][:1000]#[()]
        self.trns = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = {
            'img': self.X[idx],
            'coords': self.y[idx],
            'mask': self.masks[idx]
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
		mask = cv2.imread(mask_name,0)[:,:,None]
		
		sample = {
			'img': img,
			'hand': hand,
			'mask': mask,
			'hmap': hand.getHeatmap(img.shape)
		}
	
		if self.transform:
			sample = self.transform(sample)
		return sample


class Normalize(object):
	def __call__(self, sample):
		sample['img'] = sample['img'].astype('float32') / 255
		sample['hmap'] = sample['hmap'].astype('float32') / sample['hmap'].max()
		return sample
    
class Coords2Hmap():
	def __init__(self, sigma):
		self.sigma = sigma

	def __call__(self, sample):
		img_shape = sample['img'].shape[:2]
		hmap = np.zeros((*img_shape, 21),'float32')
		coords = sample['coords']
		hmap[coords[:,0],
			coords[:,1],
			np.arange(21)] = 10
		sample['hmap'] = cv2.GaussianBlur(hmap, (15, 15), self.sigma)
		return sample


class RotateNCrop():
	"""
	Rotate both image and heatmaps on a random angle from rotation_range,
	crop from in_size to out_size around the hand
	"""
	def __init__(self, rotation_range=(-45,45), in_size=320, out_size=128):
		self.rotation_range = rotation_range
		self.in_size = in_size
		self.out_size = out_size					  

	def __call__(self, sample):
		img, hand, heatmap = sample['img'], sample['hand'], sample['hmap']
		
		assert hand.fully_visible, "processing a hand that's not fully visible"
		cnt, rad = hand.getCircle()

		img_cnt = self.in_size // 2
		scale = 1 if rad < 70 else 70/rad
		angle = np.random.randint(*self.rotation_range)
		
		matr = cv2.getRotationMatrix2D(tuple(cnt), angle, scale)
		matr[:,2] += img_cnt - cnt
		window = slice(img_cnt - self.out_size//2,
					   img_cnt + self.out_size//2)

		block = cv2.warpAffine(
					np.c_[mask, img, heatmap], matr, #done to prevent images from rotatig differently
					(self.in_size, self.in_size),
					borderMode=cv2.BORDER_REPLICATE,
					flags=cv2.INTER_
					).astype('float32')

		sample['mask'] = block[window, window, :1]
		sample['img'] = block[window,window,1:4]
		sample['hmap'] = block[window,window,4:]
		return sample
    
class Rotate():
	"""
	Rotate both image and heatmaps on a random angle from rotation_range,
	"""
	def __init__(self, rotation_range=(-25,25)):
		self.rotation_range = rotation_range
		
	def __call__(self, sample):
		angle = random.uniform(*self.rotation_range)
		for ent in ['img', 'hmap', 'mask']:
			sample[ent] = ndimage.rotate(sample[ent], angle, reshape=False)
		return sample

class CenterNCrop():
	"""
	crop from in_size to out_size around the hand
	"""
	def __init__(self, rotation_range=(-45,45), in_size=320, out_size=128):
		self.rotation_range = rotation_range
		self.in_size = in_size
		self.out_size = out_size					  

	def __call__(self, sample):
		img, hand, heatmap, mask = sample['img'], sample['hand'], sample['hmap'], sample['mask']
		
		assert hand.fully_visible, "processing a hand that's not fully visible"
		cnt, rad = hand.getCircle()

		img_cnt = self.in_size // 2
		scale = 50/rad # to fit it into the cropped image
		
		matr = cv2.getRotationMatrix2D(tuple(cnt), 0, scale)
		matr[:,2] += img_cnt - cnt
		window = slice(img_cnt - self.out_size//2,
					   img_cnt + self.out_size//2)

		block = cv2.warpAffine(
					np.c_[mask, img, heatmap], matr, #done to prevent images from rotatig differently
					(self.in_size, self.in_size),
					borderMode=cv2.BORDER_REPLICATE).astype('float32')

		sample['mask'] = block[window, window, :1]
		sample['img'] = block[window, window, 1:4]
		sample['hmap'] = block[window, window, 4:]

		connComps = cv2.connectedComponents(
			(sample['mask']>1).astype('uint8')
			)[1]
		label = connComps[self.out_size//2, self.out_size//2]
		sample['mask'] = (connComps == label).astype('uint8')[:,:,None]
		return sample


class ToTensor(object):
	"""
	convert ndarrays in sample to Tensors
	"""
	def __call__(self, sample):
		img, hmap, mask = sample['img'], sample['hmap'], sample['mask']

		img = img.transpose((2,0,1))
		hmap = hmap.transpose((2,0,1))
		mask = mask.transpose((2,0,1))

		return {
			'img': torch.from_numpy(img).type(torch.float32),
			'hmap': torch.from_numpy(hmap).type(torch.float32),
			'mask': torch.from_numpy(mask).type(torch.float32)
		}

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