import torch
import numpy as np
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DynamicAdjustmentController:
    """
    DAC module to dynamically adjust learning rate and optimizer during training.
    """

    def __init__(self, optimizer, initial_lr=0.001, switch_optimizer_epoch=5, min_lr=1e-5, decay_factor=0.7):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.switch_optimizer_epoch = switch_optimizer_epoch
        self.epoch = 0
        self.loss_history = []
        self.optimizer_switched = False

    def update(self, loss):
        self.loss_history.append(loss)
        self.epoch += 1

        # Learning rate decay logic (based on recent plateau)
        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            if max(recent_losses) - min(recent_losses) < 0.01:
                self._decay_lr()

        # Optimizer switching (e.g., from Adam to SGD)
        if self.epoch == self.switch_optimizer_epoch and not self.optimizer_switched:
            if isinstance(self.optimizer, torch.optim.Adam):
                logger.info("🔁 DAC: Switching optimizer from Adam to SGD")
                params = self.optimizer.param_groups[0]['params']
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
                self.optimizer_switched = True
        print(f"[DAC] Epoch {self.epoch}: LR = {self.optimizer.param_groups[0]['lr']:.6f}", flush=True)
        print("🔁 DAC switching optimizer to SGD", flush=True)

    def _decay_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(self.min_lr, old_lr * self.decay_factor)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                print(f"📉 DAC reduced LR: {old_lr:.6f} ➜ {new_lr:.6f}", flush=True)

    def get_optimizer(self):
        return self.optimizer
