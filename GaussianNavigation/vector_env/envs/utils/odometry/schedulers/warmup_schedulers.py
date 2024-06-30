"""
Used code from https://github.com/Tony-Y/pytorch_warmup as an example and set group['lr'] = self.optimizer_lr * omega
instead of group['lr'] *= omega as in original implementation
(because it requires lr to be reset every time, see https://github.com/Tony-Y/pytorch_warmup/pull/1).
"""
from torch.optim import Optimizer


class BaseWarmup(object):
    """Base class for all warmup schedules

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_params, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.optimizer_lr = self.optimizer.param_groups[0]['lr']
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.dampen()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """Dampen the learning rates.

        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group['lr'] = self.optimizer_lr * omega

    def warmup_factor(self, step, **params):
        raise NotImplementedError


def get_warmup_params(warmup_period, group_count):
    if type(warmup_period) == list:
        if len(warmup_period) != group_count:
            raise ValueError(
                'size of warmup_period does not equal {}.'.format(group_count))
        for x in warmup_period:
            if type(x) != int:
                raise ValueError(
                    'An element in warmup_period, {}, is not an int.'.format(
                        type(x).__name__))
        warmup_params = [dict(warmup_period=x) for x in warmup_period]
    elif type(warmup_period) == int:
        warmup_params = [dict(warmup_period=warmup_period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} is not a list nor an int.'.format(
            type(warmup_period).__name__))
    return warmup_params


class LinearWarmup(BaseWarmup):
    """Linear warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super(LinearWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return min(1.0, (step+1) / warmup_period)


class UntunedLinearWarmup(LinearWarmup):
    """Untuned linear warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(2.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedLinearWarmup, self).__init__(optimizer, warmup_period, last_step)
