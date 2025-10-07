import jax
import jax.numpy as jnp

from .sampler import (
    get_alpha,
    get_alpha_bar,
    get_beta,
    get_beta_bar,
    get_xt,
    get_xt_1,
    loss_fn,
)
from .model import UNet
from .data import DataLoader
from .optim import MomentumSGD


class Trainer:

    def __init__(
        self,
        num_epochs: int,
        lr: float = 1e-3,
        momentum: float = 1e-4,
        alpha_bar_start: float = 1.0,
        alpha_bar_end: float = 0.0,
        num_timesteps: int = 1000,
        checkpoint_path: str = "checkpoints.npy",
    ):
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.momentum = momentum

        self.num_timesteps = num_timesteps
        self.timesteps = jnp.array(list(range(self.num_timesteps)))
        self.alpha_bar = get_alpha_bar(alpha_bar_start, alpha_bar_end, self.num_timesteps)
        self.alpha = get_alpha(self.alpha_bar)
        self.beta_bar = get_beta_bar(self.alpha_bar)
        self.beta = get_beta(self.alpha)

        self.loss_fn = jax.value_and_grad(loss_fn)

    def run(self, model: UNet, dataloader: DataLoader):

        params = model.named_parameters()
        optim = MomentumSGD(params=params, lr=self.lr, momentum=self.momentum)

        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch + 1}/{self.num_epochs}]")
            for i, batch in enumerate(dataloader):
                print(f"\tBatch [{i + 1}/{len(dataloader)}]", end=" | ")
                key = jax.random.PRNGKey()
                images, labels = batch
                batch_size = images.shape[0]

                noises = jax.random.normal(key, shape=images.shape)
                timesteps = jax.random.uniform(key, (batch_size,), minval=0, maxval=self.num_timesteps)
                alpha_bars = self.alpha_bar[timesteps]
                beta_bars = self.beta_bar[timesteps]
                xt = jax.vmap(get_xt)(images, noises, alpha_bars, beta_bars)

                preds = model(xt, timesteps)

                loss_value, loss_grad = self.loss_fn(preds, noises)

                optim.step(model.named_parameters().items(), loss_grad)

                print(f"loss value: {loss_value} | loss_grad: {loss_grad}")

        jnp.savez(self.checkpoint_path, **model.named_parameters())
        print(f"Training finished, model saved to {self.checkpoint_path}")
