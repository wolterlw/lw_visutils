import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from collections import deque

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
        return sum(self.vals)/len(self.vals)

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
        unsummed_kl = p * ((p + eps).log() - (q + eps).log())
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