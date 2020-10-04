# noinspection PyProtectedMember
from torch.optim.lr_scheduler import (_LRScheduler, MultiStepLR,
                                        CosineAnnealingLR)


# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self,
                 optimizer,
                 multiplier,
                 warmup_epoch,
                 after_scheduler,
                 last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * \
                self.last_epoch / self.warmup_epoch + 1.)
                for base_lr in self.base_lrs]

    # def step(self, epoch=None):
    #     if epoch is None:
    #         epoch = self.last_epoch + 1
    #     self.last_epoch = epoch
    #     if epoch > self.warmup_epoch:
    #         self.after_scheduler.step(epoch - self.warmup_epoch)
    #     else:
    #         super(GradualWarmupScheduler, self).step(epoch)
    def step(self, epoch=None):
        """ >= PyTorch 1.5
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(None)
        else:
            super(GradualWarmupScheduler, self).step(None)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items(
        ) if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def get_scheduler(optimizer, n_iter_per_epoch, cfg):
    policy = cfg.lr_config['policy']
    warmup_epoch = cfg.lr_config['warmup_epoch']

    if "cosine" == policy:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=(cfg.total_epochs - warmup_epoch) * n_iter_per_epoch)
    elif "step" == policy:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=cfg.lr_config['step_decay_rate'],
            milestones=[(m - warmup_epoch) * n_iter_per_epoch for m in
                        cfg.lr_config['step_decay_epochs']])
    else:
        raise NotImplementedError(
            f"scheduler {policy} not supported")

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=cfg.lr_config['warmup_multiplier'],
        after_scheduler=scheduler,
        warmup_epoch=warmup_epoch * n_iter_per_epoch)
    return scheduler
