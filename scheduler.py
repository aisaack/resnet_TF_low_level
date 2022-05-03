from abc import ABCMeta
from math import (exp, pi, cos)

class BaseScheduler(metaclass=ABCMeta):
    def __init__(
        self,
        init_lr,
        decay_steps
    ):
        self.init_lr = init_lr
        self.decay_steps = decay_steps

    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def __call__(self, step):
        pass


class ExpDecay(BaseScheduler):
    def __init__(
        self,
        init_lr,
        decay_rate,
        decay_steps
    ):
        super(ExpDecay, self).__init__(init_lr, decay_steps)
        self.decay_rate = decay_rate

    def policy(self, step):
        return self.init_lr * self.decay_rate ** exp(step / self.decay_steps)

    def __call__(self, step):
        return self.policy(step)


class CosineDecay(BaseScheduler):
    def __init__(
        self,
        init_lr,
        decay_steps,
        alpha = 0,
    ):
        super(CosineDecay, self).__init__(init_lr, decay_steps)
        self.alpha = alpha
   
    def policy(self, step):
        step = min(step, self.decay_steps)
        cos_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cos_decay + self.alpha
        return self.init_lr * decayed

    def __call__(self, step):
        return self.policy(step)


class LinearDecay(BaseScheduler):
    def __init__(
        self,
        init_lr,
        decay_steps,
        decay_rate,
        decay_cycle
    ):
        super(LinearDecay, self).__init__(init_lr, decay_steps)
        self.decay_rate = decay_rate
        self.decay_cycle = decay_cycle

    def policy(self, step):
        if step in self.decay_cycle:
            self.init_lr *= self.decay_rate
            return self.init_lr
        return self.init_lr

    def __call__(self, step):
        return self.policy(step)
