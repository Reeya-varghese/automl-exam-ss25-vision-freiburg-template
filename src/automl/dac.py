import torch
import logging

logger = logging.getLogger(__name__)

class DynamicAlgorithmController:
    """
    This function, DAC (Dynamic algorithm Controller) module dynamically adjusts the learning rate
    and switches the optimizer based on training performance.

    Args:
        optimizer (torch.optim.Optimizer): The initial optimizer (e.g., Adam).
        initial_lr (float): Starting learning rate.
        switch_optimizer_epoch (int): Epoch number at which to switch optimizers.
        min_lr (float): Minimum learning rate allowed during decay.
        decay_factor (float): Factor by which to decay the learning rate.
    """

    def __init__(self, optimizer, initial_lr=0.001, switch_optimizer_epoch=5, min_lr=1e-5, decay_factor=0.7):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.switch_optimizer_epoch = switch_optimizer_epoch
        self.epoch = 0  # Track current epoch number
        self.loss_history = []  # Store recent loss values
        self.optimizer_switched = False  # Flag to avoid multiple switches

    def update(self, loss):
        """
        This function, Updates DAC with the latest loss value, it checks if learning rate needs decay,
        and optionally switch optimizers.

        Args:
            loss (float): Latest training loss.
        """
        self.loss_history.append(loss)
        self.epoch += 1

        # Decay learning rate if recent losses show minimal improvement
        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            if max(recent_losses) - min(recent_losses) < 0.01:
                self._decay_lr()

        # Switch optimizer from Adam to SGD at a specified epoch
        if self.epoch == self.switch_optimizer_epoch and not self.optimizer_switched:
            if isinstance(self.optimizer, torch.optim.Adam):
                logger.info("DAC: Switching optimizer from Adam to SGD")
                params = self.optimizer.param_groups[0]['params']
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
                self.optimizer_switched = True

        # Print current learning rate and status
        print(f"[DAC] Epoch {self.epoch}: LR = {self.optimizer.param_groups[0]['lr']:.6f}", flush=True)
        print("DAC switching optimizer to SGD", flush=True)

    def _decay_lr(self):
        """
        This function is a Internal method to decay the learning rate if the loss plateaus.
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(self.min_lr, old_lr * self.decay_factor)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                print(f"DAC reduced LR: {old_lr:.6f} ➜ {new_lr:.6f}", flush=True)

    def get_optimizer(self):
        """
        Returns the current optimizer.
        """
        return self.optimizer
