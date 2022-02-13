from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer, num_iters, warmup_iters, last_epoch=-1):
        self.total_iters = num_iters + 0.0
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_step = warmup_iters
        if self.warmup_step is None:
            self.warmup_step = 0.0
        assert self.warmup_step >= 0, 'warm up step must be >= 0'
        super(WarmupCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_step > 0:
            if self.last_epoch <= self.warmup_step:
                return [base_lr / self.warmup_step * self.last_epoch for base_lr in self.base_lrs]
        return [base_lr * 0.5*(1 + math.cos((self.last_epoch - self.warmup_step
                                                  ) / (self.total_iters - self.warmup_step) * math.pi))
                for base_lr in self.base_lrs]
