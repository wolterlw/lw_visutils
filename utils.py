import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from collections import deque
from torch.optim import Optimizer

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

class Saver():
    def __init__(self, params_path='./', better=lambda x,y: x>y):
        self.prev_best = 0
        self.better = better
        self.params_path = params_path
    
    def save(self, model, metric):
        if self.better(metric, self.prev_best):
            self.prev_best = metric
            torch.save(model.state_dict(),
                       self.params_path + f"weights_{metric:.4f}.pt")

class TBhook():
    def __init__(self, launch_dir):
        self.writer = SummaryWriter(launch_dir)
        
    def write_graph(self, model):
        zeros = torch.zeros(1,3,128,128,dtype=torch.float32).cuda()
        self.writer.add_graph(model, zeros, verbose=False)
        
    def write_scalars(self, loss, metrics, global_step):
        self.writer.add_scalars('data/metrics', metrics, global_step)
        self.writer.add_scalars('data/losses', loss, global_step)        
    
    def write_img(self, img, name='test_image', step=0):
        self.writer.add_image('images/'+name, img, step)
        
    def write_hmaps(self, hmaps, name='test_hmap', step=0):
        grid = vutils.make_grid(hmaps.view(-1,1,128,128), normalize=True)
        self.writer.add_image('images/'+name, grid, step)

    def close(self):
        self.writer.close()

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
        return self._js(y_pred, y_true)

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
        self.colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
    
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

class HeatmapBatch(nn.Module):
    """Module that converts coordinates to heatmaps on the GPU"""
    __constants__ = ['idx', 'kernel', 'bg']

    def __init__(self, batch_size=1, hmap_size=128, sigma=1.5, kernel_size=9, kp_num=21):
        super(HeatmapBatch, self).__init__()
        
        self.pad = kernel_size//2
        
        kernel0 = cv2.getGaussianKernel(kernel_size, sigma)
        kernel0 = torch.from_numpy(kernel0 @ kernel0.T)
        kernel = torch.zeros(kp_num,kp_num,kernel_size,kernel_size)
        for i in range(kp_num):
            kernel[i,i] = kernel0
        
        bg = torch.zeros(batch_size, kp_num, hmap_size, hmap_size)
        idx = torch.arange(batch_size*kp_num)
        
        self.register_buffer('idx', idx.long())
        self.register_buffer('kernel', kernel.float())
        self.register_buffer('bg', bg.float())

    def forward(self, x):
        crd = x.view(-1,2)
        self.bg.zero_()
        self.bg.flatten(0,1)[self.idx, crd[:,0], crd[:,1]] = 10
        res = F.conv2d(self.bg, self.kernel, padding=self.pad)
        return res
