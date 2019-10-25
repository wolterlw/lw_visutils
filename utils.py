import torch
import torchvision.utils as vutils
from collections import deque
from torch.optim import Optimizer
import torch.nn as nn
from cv2 import getGaussianKernel
import torch.nn.functional as F

class AverageMeter(object):
    """Keeps an average of a subsequence of certain size
    
    Attributes:
        max_len (int): maximal squence length
    """
    
    def __init__(self, max_len=50):
        self.vals = deque(maxlen=max_len)
        self.max_len = max_len
    
    def update(self, val, n=1):
        self.vals.extend([val]*n)

    def avg(self):
        return sum(self.vals)/len(self.vals) if len(self.vals) else 0

class DivergenceLoss():

    """Jensen-Shannon Loss class
    
    Attributes:
        reduction (string): mean or sum
    """
    
    eps = 1e-24
    def __init__(self, reduction='mean'):
        self.red = reduction

    def _kl(self, p, q):
        unsummed_kl = p * ((p + self.eps).log() - (q + self.eps).log())
        if self.red == 'mean':
            kl_vals = unsummed_kl.mean()
        else:
            kl_vals = unsummed_kl.sum()
        return kl_vals

    def _js(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * self._kl(p, m) + 0.5 * self._kl(q, m)

    def __call__(self, y_pred, y_true):
        """Calculates JS divergence
        
        Args:
            y_pred torch.floatTensor: predicted heatmaps
            y_true torch.floatTensor: ground truth heatmaps
        
        Returns:
            torch floatTensor: calculated JS divergence
        """
        return self._js(F.relu(y_pred), F.relu(y_true))

class HandViz():
    def __init__(self, joint_array, img_dim=128):
        self.array = np.clip(joint_array[:,:2],0,img_dim).astype('int')
        self.joints = {
            'wrist': self.array[:1], # black (it's a dot anyways)
            'thumb': self.array[1:5], # red
            'index': self.array[5:9], # green
            'middle': self.array[9:13], # blue
            'ring': self.array[13:17], # cyan
            'pinky': self.array[17:21], # magenta
        }
        self.colors = [(  0,  0,  0),(255,  0,  0),(  0,255,  0),
                       (  0,  0,255),(  0,255,255),(255,  0,255)]
    
    def draw(self, img):
        assert type(img) is np.ndarray
        assert len(img.shape) == 2, f"{img.shape}" # input - grayscale
        res_img = np.stack([img]*3,axis=-1)
        
        for joint, color in zip(self.joints.values(), self.colors):
            for beg,end in zip(joint, joint[1:]):
                cv2.line(res_img, tuple(beg),tuple(end), color, 1)
            cv2.line(res_img, tuple(self.joints['wrist'][0]),
                              tuple(joint[0]), color, 1)
        return res_img

class ImgVisualizer():
    def __init__(self, writer):
        self.writer = writer

    @staticmethod
    def combine_hmaps(hmap_gt, hmap0, hmap1):
        stacked = torch.stack([hmap_gt, hmap0, hmap1], dim=0).transpose(0,1)
        grid = vutils.make_grid(
            stacked, normalize=True, 
            nrow=7, pad_value=1
        ).numpy().transpose(1,2,0)
        return grid

    def write(self, img, hmap_gt, hmap_pred, step, img_idx=0, valid=False):
        gt = hmap_gt[img_idx].detach().squeeze().cpu()
        hmap0 = hmap_pred['hmap0'][img_idx].squeeze().detach().cpu()
        hmap1 = hmap_pred['hmap1'][img_idx].squeeze().detach().cpu()
        inp_img = img[img_idx].detach().cpu()

        grid = self.combine_hmaps(gt, hmap0, hmap1)

        log = {
            "img": [self.writer.Image(inp_img)],
            "heatmaps": [self.writer.Image(grid)]
            }
        if valid:
            log = {k + "_val": v for k,v in log.items()}
        self.writer.log(log)

class HeatmapBatch(nn.Module):
    """Module that converts coordinates to heatmaps on the GPU"""
    __constants__ = ['idx', 'kernel', 'bg']

    def __init__(self, batch_size=1, hmap_size=128, 
                sigma=1.5, kernel_size=9, kp_num=21):
        super(HeatmapBatch, self).__init__()
        
        self.pad = kernel_size//2
        
        kernel0 = getGaussianKernel(kernel_size, sigma)
        kernel0 = torch.from_numpy(kernel0 @ kernel0.T)
        self._gauss_val = 1 / kernel0.max()
        kernel = torch.zeros(kp_num,kp_num,kernel_size,kernel_size)
        for i in range(kp_num):
            kernel[i,i] = kernel0

        
        bg = torch.zeros((batch_size, kp_num, hmap_size, hmap_size))
        idx = torch.arange(batch_size*kp_num)
        
        self.register_buffer('idx', idx.long())
        self.register_buffer('kernel', kernel.float())
        self.register_buffer('bg', bg.float())
        self.noise_mean = 0
        self.noise_std = 0.01

    def set_noise(self, mean, std):
        self.noise_mean = mean
        self.noise_std = std

    def forward(self, x):
        crd = x.view(-1,2)
        self.bg.normal_(self.noise_mean, self.noise_std)
        self.bg.flatten(0,1)[self.idx, crd[:,1], crd[:,0]] = self._gauss_val
        res = F.relu(F.conv2d(self.bg, self.kernel, padding=self.pad))
        return res
