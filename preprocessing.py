import os
import pickle
import pathlib

import numpy as np
import cv2
from skimage import io #TODO: delete 
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
            self.all_coords = [self._read_joints(x) for x in self.joint_path]
        
        
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

class Normalize(object):
	def __init__(self, hmap=True):
		self.hmap = hmap

	def __call__(self, sample):
		mean = np.r_[[0.485, 0.456, 0.406]]
		std  = np.r_[[0.229, 0.224, 0.225]]
		sample['img'] = (sample['img'].astype('float32') / 255 - mean) / std
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
	def __init__(self, rotation_range=(-25,25), keys=['img','hmap','mask']):
		self._keys = keys
		self.rotation_range = rotation_range
		
	def __call__(self, sample):
		angle = random.uniform(*self.rotation_range)
		for ent in self._keys:
			sample[ent] = np.abs(ndimage.rotate(sample[ent], angle, reshape=False))
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