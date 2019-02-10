import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from collections import deque

class AverageMeter(object):
    """Computes and stores the average and current value"""
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