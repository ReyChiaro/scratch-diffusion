import jax.lax as lax
import jax.numpy as jnp

from jax import Array


def get_alpha_bar(start: float, end: float, timesteps: int = 1000) -> Array:
    alpha_bar = jnp.linspace(jnp.array(start), jnp.array(end), timesteps, endpoint=True)
    alpha_bar = jnp.concatenate([jnp.array(1.0), alpha_bar], axis=0)
    return alpha_bar


def get_alpha(alpha_bar: Array) -> Array:
    return jnp.array([alpha_bar[i] / alpha_bar[i - 1] for i in range(alpha_bar.shape[0])])


def get_beta_bar(alpha_bar: Array) -> Array:
    return lax.sqrt(1.0 - alpha_bar**2)


def get_beta(alpha: Array) -> Array:
    return lax.sqrt(1.0 - alpha**2)


def get_xt(x0: Array, noise: Array, alpha_bar: Array, beta_bar: Array) -> Array:
    return alpha_bar * x0 + beta_bar * noise


def get_xt_1(xt: Array, noise: Array, alpha: Array, beta: Array) -> Array:
    return (xt - beta * noise) / alpha


def loss_fn(preds: Array, targets: Array) -> Array:
    return jnp.mean((preds - targets) ** 2)
