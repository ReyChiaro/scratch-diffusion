import jax.numpy as jnp


class MomentumSGD:

    def __init__(self, params: dict, lr: float, momentum: float):

        self.lr = lr
        self.momentum = momentum

        self.vs = {k: jnp.zeros_like(v) for k, v in params}

    def step(self, params: dict, grads: dict):
        for k in params.keys():
            v = self.vs[k]
            g = grads[k]

            v = self.momentum * v + (1.0 - self.momentum) * g
            p = self.params[k] - self.lr * v

            self.params[k] = self.params[k] + p
            self.vs[k] = v
