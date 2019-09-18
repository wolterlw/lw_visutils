import re
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from tqdm import tqdm
from torch.utils.data import Dataset


# class Hand():
# 	"""
# 	Helper class used to visualize hand joints and generate gaussian heatmaps
# 	"""
# 	colormap = {
# 			'wrist': 'w',
# 			'thumb': '#ffc857',
# 			'index': '#e9724c',
# 			'middle': '#c5283d',
# 			'ring': '#00fddc',
# 			'pinky': '#255f85',
# 		}

# 	def __init__(self, joint_array, img_idx):
# 		self.img_idx = img_idx
# 		self.fully_visible = (joint_array > 0).all()
# 		self.array = np.clip(joint_array[:,:2],0,319)
# 		self.joints = {
# 			'wrist': self.array[:1],
# 			'thumb': self.array[1:5],
# 			'index': self.array[5:9],
# 			'middle': self.array[9:13],
# 			'ring': self.array[13:17],
# 			'pinky': self.array[17:21],
# 		}
	
# 	def draw(self, axis):
# 		for k in self.joints.keys():
# 			axis.plot(
# 				self.joints[k][:,0],
# 				self.joints[k][:,1],
# 				c=self.colormap[k])
# 			axis.plot([self.joints['wrist'][0,0],self.joints[k][-1,0]],
# 					 [self.joints['wrist'][0,1],self.joints[k][-1,1]],
# 					 c=self.colormap[k])
			
# 	def getHeatmap(self, img_shape):
# 		heatmap = np.zeros((*img_shape[:2], 21),'float')
# 		heatmap[self.array[:,1].astype('uint'),
# 				self.array[:,0].astype('uint'),
# 				np.arange(21)] = 10
# 		heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1)
# 		heatmap /= heatmap.max()
# 		return heatmap
		
# 	def getCircle(self):
# 		min_ = self.array.min(axis=0)
# 		max_ = self.array.max(axis=0)
# 		center = min_ + (max_ - min_) / 2
# 		radius = np.sqrt(((max_ - center)**2).sum())
# 		return center, radius

# =============================== DATASET WRAPPERS ==================================

class HDFDataset():
	def __init__(self, file_path, transform=None):
		hdf_file = h5py.File(file_path, 'r')
		self.imgs = hdf_file['img'][()]
		self.coords = hdf_file['coords'][()]
		self.trns = transform
		hdf_file.close()
	
	def __len__(self):
		return len(self.imgs)
	
	def __getitem__(self, idx):
		sample = {
			'img': self.imgs[idx],
			'coords': self.coords[idx],
		}
		return self.trns(sample) if self.trns else sample

class GoogleGlassDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		root_dir = Path(root_dir).expanduser()

		self.mask_files = sorted(
			root_dir.glob("_L*/*/mask*.jpg")
		)
		self.frame_files = sorted(
			root_dir.glob("_L*/*/frame*.jpg")
		)
		
		assert len(self.frame_files) == len(self.mask_files), "whoops, something's wrong"
		self.trns = transform
		
	def __len__(self):
		return len(self.frame_files)
	
	def __getitem__(self, idx):
		sample = {
			'img': self.frame_files[idx],
			'mask': self.mask_files[idx]
		}
		return self.trns(sample) if self.trns else sample

class EYTHDataset(Dataset):
	"""Wrapper for Ego YouTube dataset"""
	def __init__(self, root_dir, transform=None):
		root_dir = Path(root_dir).expanduser()

		self.img_files = sorted(
			root_dir.glob("images/*/*.jpg")
		)
		self.mask_files = sorted(
			root_dir.glob("masks/*/*.png")
		)
		
		assert len(self.img_files) == len(self.mask_files), "whoops, something's wrong"
		self.trns = transform
		
	def __len__(self):
		return len(self.img_files)
	
	def __getitem__(self, idx):
		sample = {
			'img': self.img_files[idx],
			'mask': self.mask_files[idx]
		}
		return self.trns(sample) if self.trns else sample

class RHDDataset(Dataset):
    """
    Versatile RHD dataset abstraction used for preprocessing
    """
    def __init__(self, root_dir, dset='training', transform=None):
        super(RHDDataset, self).__init__()

        self.root_dir = Path(root_dir).expanduser()
        subsets = [x.name for x in self.root_dir.glob("*") if x.is_dir()] 
        assert dset in subsets, f"dset shoud be one of {subsets}"
        self.root_dir /= dset

        with open(next(self.root_dir.glob("anno*pickle")),'rb') as f:
            anno = pickle.load(f)
        # unpacking hands into separate records
        # only leaving hands that are fully inside the image
        hands_valid = [
            x for i,y in anno.items()\
            for x in self._split_hands(y, i) if self._valid_hand(x)]
        
        # changing back to dict for lookup efficiency
        self.anno = {
            i: x for i,x in enumerate(hands_valid)
        }
        self.transform = transform

    def __len__(self):
        return len(self.anno)*2
    
    @staticmethod
    def _get_bbox(coords, pad=10):
        min_c  = coords.min(axis=0) - pad
        max_c = coords.max(axis=0) + pad
        return np.clip(np.r_[min_c, max_c].astype('int'), 0, 320)
    
    def _split_hands(self, x, index):
        return [{
                'filename': f"{index:05}.png",
                'K': x['K'],
                'xyz': x['xyz'][slc_own], 
                'uv_own': x['uv_vis'][slc_own,:2],
                'box_own': self._get_bbox(
                    x['uv_vis'][slc_own,:2]
                ),
                'box_other': self._get_bbox(
                    x['uv_vis'][slc_other,:2]
                )}\
                for slc_own, slc_other in 
                            zip((slice(0,21), slice(21,42)),
                                (slice(21,42), slice(0,21)))
            ]
    @staticmethod
    def _valid_hand(x):
        return (x['uv_own'] >= 0).all() and (x['uv_own'] < 320).all() 

    def __getitem__(self, idx):
        record = self.anno[idx] 
        
        img_color = (self.root_dir / "color" / record['filename']).as_posix()
        img_depth = (self.root_dir / "depth" / record['filename']).as_posix()
        img_mask = (self.root_dir / "mask" / record['filename']).as_posix()

        sample = {
            'img': img_color,
            'mask': img_mask,
            'depth': img_depth,
        }
        sample.update(record)

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
		self.root_dir = Path(root_dir).expanduser()
		
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
		image = self.image_path[idx]
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
		
		self.root_dir = Path(root_folder).expanduser()
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
		img = self.img_files[idx]
		points = self._project(self.joints3D[idx])[:,::-1]
		points = points[self.reidx_coords,:]

		sample = {'img': img, 'coords': points}
		return self.transforms(sample) if self.transforms else sample
