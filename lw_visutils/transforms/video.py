import numpy as np
import cv2

def AffineNCropImg(img_source, M_source, target_res=(256, 256), keep_original=False):
    r"""performs an affine transformation on src only
    Affine transformation M should be present in the sample at M_source.
    """
    assert isinstance(target_res, tuple)

    def f(sample):
        M = sample[M_source][:2]
        sample[img_source] = cv2.warpAffine(sample[img_source], M, target_res)
        return sample
    return f


def GetAffineTransform(target_res=(256, 256)):
    # T = np.eye(3)
    R = np.eye(3)
    cnt = tuple([x // 2 for x in target_res])
    res = target_res

    def _get_translation(center, scale):
        """borrowed from original code to process scale properly"""
        h = 200 * scale
        T = np.eye(3)
        T[0, 0] = res[1] / h
        T[1, 1] = res[0] / h
        T[0, 2] = res[1] * (-center[0] / h + .5)
        T[1, 2] = res[0] * (-center[1] / h + .5)
        return T

    def f(sample):
        T = _get_translation(sample['center'], sample['scale'])
        R = cv2.getRotationMatrix2D(cnt, sample['rotation'], 1)

        M = R @ T
        M3x3 = np.eye(3, dtype='float32')
        M3x3[:2] = M
        sample['M'] = M3x3

        return sample
    return f

def GetCenterScale(dest_size=300):
    def f(sample):
        sample['scale'] = sample['size'] / dest_size
        return sample
    returnf 

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Cache(metaclass=Singleton):
    def __init__(self,):
        self.kps = None
        self.shape = None
        self.size = None
        self.non_normal = 0
        self.backup_kp = None

    def reset(self,):
        self.kps = None
        self.shape = None
        self.size = None

    def keypoints_ok(self, kps):
        max_size = max(kps.max(axis=0) - kps.min(axis=0))
        ratio = max_size / self.size
        not_ok = (kps < 0).any() or\
                 (kps[:,0] > self.shape[0]*1.2).any() or\
                 (kps[:,1] > self.shape[1]*1.2).any() or\
                 ratio < 0.4
        return not not_ok
        
    def __call__(self, sample):
        if self.shape is None:
            self.shape = sample['img'].shape[:2]
            self.size = max(self.shape)
        if 'keypoints_orig' in sample:
            kps = sample['keypoints_orig']
            if self.keypoints_ok(kps):
                self.kps = kps
                if self.backup_kp is None:
                    self.backup_kp = kps
            elif not self.backup_kp is None:
                self.kps = 0.5 * self.kps + 0.5 * self.backup_kp
            else:
                self.kps = kps
        elif not self.kps is None:
            sample['keypoints'] = self.kps
        return sample

class TrackCenterScale():
    def __init__(self, margins_min=20, margin_soft=30):
        self.mm = margins_min
        self.ms = margin_soft
        
        self.shift_center = np.r_[0,-30]
        self.scale_coef = 1.9
        
        self.sqr_old = None
        
        self.records = []
        
        self.const = 200 / 256
    
    def kps_to_sqr(self, keypoints):
        xy_max = keypoints.max(axis=0)
        xy_min = keypoints.min(axis=0)
        center = (xy_max + xy_min) / 2  + self.shift_center
        size = (xy_max - xy_min).max() * self.scale_coef
        return xy_max, xy_min, center, size

    def tight2square(self, center, size):
        h = size / 2 * self.const
        xy_min = center - h
        xy_max = center + h
        return xy_max, xy_min
        
        
    def __call__(self, sample):
        keypoints = sample['keypoints']
        kps_vis = keypoints[keypoints[:,2]>0, :2]
        
        xy_max,xy_min,center,size = self.kps_to_sqr(kps_vis)
        xy_max_loose, xy_min_loose = self.tight2square(center, size)
        
        sample['sqr_tight'] = (xy_max,xy_min)
        sample['sqr_loose'] = (xy_max_loose, xy_min_loose)
        
        if self.sqr_old is None:
            self.sqr_old = (xy_max_loose, xy_min_loose, center, size)
            
            sample['center'] = center
            sample['size'] = size
        else:
            xy_max_old, xy_min_old, center_old, size_old = self.sqr_old
            margin_max = xy_max_old - xy_max # loose - tight
            margin_min = xy_min - xy_min_old # loose - tight
            size_diff = abs(size_old - size) / max(0.1, min(size_old, size))
            
            margin_max_l = abs(xy_max_old - xy_max_loose)
            margin_min_l = abs(xy_min_old - xy_min_loose)
            
            
            if (margin_max < self.mm).any() or (margin_min < self.mm).any():
                xy_max_loose_new = 0.5 * xy_max_old + 0.5 * xy_max_loose
                xy_min_loose_new = 0.5 * xy_min_old + 0.5 * xy_min_loose
                center_new = 0.5 * center_old + 0.5 * center
                size_new = 1.1 * (0.5 * size_old + 0.5 * size)
                self.sqr_old = (xy_max_loose_new, xy_min_loose_new, center_new, size_new)
                sample['center'] = center_new
                sample['size'] = size_new
            elif (margin_max_l > self.ms).any() or (margin_min_l > self.ms).any():
                xy_max_loose_new = 0.7 * xy_max_old + 0.3 * xy_max_loose
                xy_min_loose_new = 0.7 * xy_min_old + 0.3 * xy_min_loose
                center_new = 0.7 * center_old + 0.3 * center
                size_new = 0.7 * size_old + 0.3 * size
                self.sqr_old = (xy_max_loose_new, xy_min_loose_new, center_new, size_new)
                sample['center'] = center_new
                sample['size'] = size_new
            else:
                sample['center'] = center_old
                sample['size'] = size_old
        
        return sample
