## References

[Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929v2)


## Review

Diffusion modeling aims to construct a sampler of a target distribution $p^*$. The idea is not to learn the mapping between a easy-to-sample distribution $q$, but construct a reverse sampler on a sequence of distributions $\{ p_{k\Delta t} \}_{k=0}^{k=1/{\Delta t}}$ (suppose we use continuous timesteps $t\in [0,1], T\Delta t = 1$, where $T$ is the total sample steps).

- Sequence of distributions: these distributions interpolate between the easy-to-sample distribution and the target distribution such that $p_1=q, p_0=p^*$. (The symbol may be different from some papers)
- Gaussian diffusion: choose $q=\mathcal{N}(0,\sigma_q^2)$, then we can construct the sequence of distributions by add Gaussian noise with a **variance exploding (VE)** method, namely $x_{t+\Delta t} = x_t + \eta_t, \eta_t \sim \mathcal{N}(0, \sigma^2), t\in \{0,\Delta t, 2\Delta t, \cdots (T-1)\Delta t\}$, or $x_{t} = x_0 + \mathcal{N}(0, \sigma_t^2)$.
  - To remain $\sigma_q$ the same, we choose $\sigma = \sqrt{\Delta t} \sigma_q$ or $\sigma_t = \sqrt{t} \sigma_q$. 
- $\sigma$s are small: if the variances added on the distributions are small enough, then the neighbor distributions are close in some degree. It means the posterior $p(x_{t-1} | x_t)$ is itself close to Gaussian $p(x_{t-1} | x_t) \approx \mathcal{N}(x_{t-1}, \sigma^2)$.
  - As the $\sigma$s are pre-defined, the only thing we want to learn is the mean of Gaussian.
- Learn $\mu$: $\mu(x_{t-1} | x_{t}) = \mathbb{E}[x_{t-1} | x_t]$, which can be converted into the regression optimization problem $\arg \max_{f} \mathbb{E}[\| f(x_t) - x_{t-1} \|_2^2]$.

## DDPM: Stochastic Sampling

