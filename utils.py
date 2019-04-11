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

class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = int(1 + self.last_batch_iteration / (2 * step_size))
        x = abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * max(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs