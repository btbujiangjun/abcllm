import torch

class LinearWarmupLinearDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * max(0.0, (1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
                for base_lr in self.base_lrs
            ]
