from functools import lru_cache
from random import randint
from warnings import showwarning

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

def OtherHandMasker():
    def f(sample):
        box_other = sample['box_other']
        box_own = sample['box_own']
        
        mask = np.ones_like(sample['img'][:,:,0])
        mask[
            box_other[1]:box_other[3],
            box_other[0]:box_other[2]
        ] = 0
        # making sure current hand is not obscured
        mask[
            box_own[1]:box_own[3],
            box_own[0]:box_own[2]
        ] = 1
        sample['img'] = sample['img'] * mask[:,:,None]
        return sample
    return f

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

class AffineTransform():
    def __init__(self, img_size=(256,256),
                 scale_min=0.8, scale_max=1.3,
                 translation_max=45):
        self.tmax = translation_max
        self.scale = (int(scale_min*10), int(scale_max*10))
        self.img_size = img_size
        self.center = (img_size[0]//2, img_size[1]//2)
        self.crd_max = max(img_size)-1
    
    @staticmethod
    def _pad1(M):
        return np.pad(
            M, ((0,0),(0,1)),
            mode='constant', 
            constant_values=1)

    def __call__(self, sample):
        M = cv2.getRotationMatrix2D(
                self.center, randint(-90,90),
                randint(*self.scale) / 10)
        M[:,2:] += np.random.uniform(-self.tmax, self.tmax, (2,1))
        sample['img'] = cv2.warpAffine(
            sample['img'],
            M, self.img_size,
            borderMode=cv2.BORDER_REFLECT)
        Mpad = self._pad1(M.T)
        Mpad[:2,2:] = 0
        
        crd_t = self._pad1(sample['coords']) @ Mpad
        sample['coords'] = np.clip(crd_t[:,:2], 1, self.crd_max)
        return sample

class AddBboxCentered():
    def __init__(self, kp_center=8, scale=1.1):
        self.kp_cnt = 12
        self.scale = 1.1
        self.b_vec = np.r_[[
            [-1,-1],
            [ 1,-1],
            [ 1, 1],
            [-1, 1]
        ]]
        
    def __call__(self, sample):
        crd = sample['coords'].astype('float32')
        center = crd[self.kp_cnt]
        max_dist = np.linalg.norm(crd - center, axis=1).max()
        box = center + self.b_vec * max_dist * self.scale
        sample['bbox'] = box.astype('float32')
        return sample

class CropByBbox():
    def __init__(self, out_size=256):
        assert type(out_size) is int
        self.out_size = (out_size, out_size)
        self.target = np.r_[[
            [       0,        0],
            [out_size,        0],
            [out_size, out_size],
            [0       , out_size]
        ]].astype('float32')
    
    @staticmethod
    def warpCoords(crd, M):
        crd_pad = np.pad(crd, ((0,0), (0,1)), constant_values=1)
        M_pad = np.pad(M, ((0,1), (0,0)))
        M_pad[2,2] = 1
        crd_new = crd_pad @ M_pad.T
        assert (crd_new[:,2] == 1).all()
        return crd_new[:,:2]
    
    def __call__(self, sample):
        src = sample['bbox'].astype('float32')
        M = cv2.getAffineTransform(src[:3], self.target[:3])
        img_new = cv2.warpAffine(sample['img'], M, self.out_size)
        crd_new = self.warpCoords(sample['coords'], M)

        if (crd_new < 0).any() or (crd_new > self.out_size).any():
            showwarning("coordinates our of bounds", RuntimeWarning, "transformers.py", 176)
        
        sample['img'] = img_new
        sample['coords'] = crd_new
        sample['bbox'] = self.target
        return sample

class GoogleStyleBbox():
    def __init__(self, out_size, box_enlarge=1.5, rand=False):
        self.out_size = out_size
        self.box_enlarge = box_enlarge
        self.R90 = np.r_[[[0,1],[-1,0]]]
        half = out_size // 2
        self._target_triangle = np.float32([
                        [half, half],
                        [half,    0],
                        [   0, half]
                    ])
        self.rand = rand
        
    def get_triangle(self, kp0, kp2, dist=1):
        """get a triangle used to calculate Affine transformation matrix"""

        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)

        dir_v_r = dir_v @ self.R90.T
        return np.float32([kp2, kp2+dir_v*dist, kp2 + dir_v_r*dist])
    
    @staticmethod
    def triangle_to_bbox(source):
        # plain old vector arithmetics
        bbox = np.c_[
            [source[2] - source[0] + source[1]],
            [source[1] + source[0] - source[2]],
            [3 * source[0] - source[1] - source[2]],
            [source[2] - source[1] + source[0]],
        ].reshape(-1,2)
        return bbox

    def __call__(self, sample):
        """
        Input {'img': (*in_shape,3), 'coords': (21,2), *}
        Output {'img': (out_size,out_size,3), 'coords': (21,2), *}
        """
        img = sample['img']
        coords_0 = sample['coords'][:,::-1].astype('float32')
        
        height = np.linalg.norm(coords_0[12] - coords_0[0])
        width = np.linalg.norm(coords_0[8] - coords_0[20])
        big_pink = np.linalg.norm(coords_0[4] - coords_0[20])
        side = max(height, width, big_pink)

        source = self.get_triangle(coords_0[0], coords_0[12], side * self.box_enlarge)

        sample['bbox'] = self.triangle_to_bbox(source)
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
        self._coords = False
        if 'coords' in keys:
            keys.remove('coords')
            self._coords = True
        self._keys = keys

    def __call__(self, sample):
        for k in self._keys:
            x = sample[k].transpose((2,0,1))
            sample[k] = torch.from_numpy(x.astype('float32'))
        if self._coords:
            sample['coords'] = torch.from_numpy(sample['coords'].astype('float32'))
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

class RandomizeCoords():
    def __init__(self, hmap_shape, noise_max=10):
        self._idx = np.arange(hmap_shape[0])
        self._max = hmap_shape[1] * 2
        self._noise = noise_max
        self._noise_shape = (hmap_shape[0], 2)

    def __call__(self, sample):
        op = np.random.randint(3)
        sample['coords'] = sample['coords'].astype('float32')
        if op == 0:
            np.random.shuffle(self._idx)
            sample['noisy_coords'] = sample['coords'][self._idx]
            return sample
        if op == 1:
            noise = np.random.uniform(-self._noise, self._noise, 
                self._noise_shape)
            sample['noisy_coords'] = np.clip(
                sample['coords'] + noise,
                0, self._max).astype('float32')

            return sample
        else:
            sample['noisy_coords'] = np.random.uniform(
                0, self._max, self._noise_shape).astype('float32')
            return sample

class RemapKeys():
    def __init__(self, map_=[('img','img'), ('coords', 'coords')]):
        assert type(map_) is list, "map_ should be in form [(from_key, to_key),]"
        assert type(map_[0]) is tuple, "map_ should be in form [(from_key, to_key),]"
        self._map = map_

    def __call__(self, sample):
        return {k_new: sample[k_old] for k_old, k_new in self._map}
