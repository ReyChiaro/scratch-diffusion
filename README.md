# scratch-diffusion
A repository for understanding, implementing denoise-diffusion/flow-matching models.

You can check the [blog](https://reychiaro.github.io/2025/07/04/paper-read/generation/) for better rendering.

## References

[Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929v2)


## Overview of Diffusions

> The goal of generative models is constructing a **sampler** for an unknown distribution $p^{*}(x)$. As long as the sampler is contructed, then we can generate **new** samples from the distribution.
> As the distribution is unknown, we should get a set of representative samples from the unknown distribution to estimate it.

We can directly learn a transformation between the unknown distribution and a simple-to-sample distribution such as Gaussian distribution. However this may be untractable as the target unknown distribution can be very large and complicated.
Diffusion models provide a general framework for learning such transformations, reducing the problem of sampling from distribution $p^{*}(x)$ to a sequence of easier sampling problems.


### Gaussian Diffusions

Let random variable $x_0 \sim p^{*} \in \mathbb{R}^d$, then construct a sequence of random variables $\{ x_1, x_2, \cdots, x_T \}$ by successively adding independent Gaussian noise with some small scale $\sigma$
$$
x_{t+1} = x_t + \eta_t, \eta_t \sim \mathcal{N}(0, \sigma^2)
$$
which is called the *forward process*. Let $\{p_t\}_{t \in \{0,1,\cdots,T\}}$ be the marginal distribution of each $x_t$. We have $\lim_{T\rightarrow \infty} p_T = \mathcal{N}(0, \sigma_q^2)$.
In practice, $T$ is sufficiently large and we can assume that $p_T \approx \mathcal{N}(0, \sigma_q^2)$, where $\sigma_q$ is the standard deviation. What we *want to* do is to learn a transformation such that given the marginal distribution $p_t$, we can produce $p_{t-1}$. And if the final marginal distribution $p_T$ is given, we can produce $p_0=p^{*}$ iteratively. This process is called *reverse process*, and the method to implemente this process is called *reverse sampler*.

The reverse process can be represented by conditional probability. At time step $t$, given the input $z$ sampled from $p_t$, the output of the reverse sampler generate a sample from the conditional distribution
$$
p(x_{t-1}|x_t=z).
$$

However, though this implies that a generative model should learn the conditional distribution for every $x_t$ which could be complicated. But we have the following insight, which will be proved in <u>DDPM sampler</u> section:

> Fact 1 (Diffusion Reverse Process). For small $\sigma$ and the Gaussian diffusion process $x_{t+1} = x_t + \eta_t$, where $\eta_t \sim \mathcal{N}(0,\sigma^2)$, then the conditional distribution $p(x_{t-1} | x_t)$ is itself close to Gaussian.
> For all time step $t$ and condition $z \in \mathbb{R}^d$, there exists some mean parameter $\mu_{t-1} \in \mathbb{R}^d$ such that $p(x_{t-1} | x_t) \approx \mathcal{N}(x_{t-1}; \mu_{t-1}, \sigma_{t-1}^2)$.

Given this fact, we find that if $\sigma_t$s are provided, then the only thing the model should learn is the mean of the distribution $p(x_{t-1}|x_t)$, which is noted as $\mu(x_{t-1}|x_t)$. Fortunately, learning the mean is much simpler than learning $p(x_{t-1}|x_t)$ as this can be seen as a regression problem: Given the joint distribution $(x_{t-1},x_t)$, from which we can easily to sample, and the definition of the mean of the distribution $p(x_{t-1}|x_t)$:
$$
\mu(x_{t-1} | x_t) \coloneqq \mathbb{E}[x_{t-1} | x_t]
$$
we have
$$
\mu(x_{t-1}) = \argmin_{f:\mathbb{R}^d\rightarrow \mathbb{R}^d} \mathbb{E}_{x_t,x_{t-1}}[\| f(x_t) - x_{t-1}\|_2^2] \\
\Rightarrow \mu(x_{t-1}) = \argmin_{f:\mathbb{R}^d\rightarrow \mathbb{R}^d} \mathbb{E}_{x_{t-1}, \eta_t}[\| f(x_{t-1} + \eta_t) - x_{t-1}\|_2^2]
$$
where the expectation is taken over sample $x_0$ from the target distribution $p^*$.

Now, the problem of learning to sample from an arbitrary distribution is converted into optimizing the regression problem.


### Abstract of Diffusions

We abstract the Gaussian settings of diffusions. Given a set of samples extracted from target distribution $p^*$, and the easy-to-sample base distribution $q$ (e.g. Gaussian or multinomial), we try to construct a sequence of distributions $p_0,p_1,\cdots, p_T$ which interpolate between $p^*$ and $q$, such that $p_0=p^*, p_T=q$ and the adjacent distribution $(p_{t-1},p_t)$ are marginally close. The aim is to learn a reverse sampler to transform distribution $p_t$ to $p_{t-1}$.

The *reverse sampler* at step $t$ is a potentially stochastic function $F_t$ such that if $x_t \sim p_t$, then the marginal distribution of $F_t(x_t)$ is exactly $p_{t-1}$, which means the reverse sampler is used to transform the distribution $p_t$ to $p_{t-1}$:
$$
\{ F_t(z): z\sim p_t \} \equiv p_{t-1}
$$


### Dicretization

What is exactly mean about *close* between $p_{t}$ and $p_{t-1}$?

Assuming we have a time-evolving function $p(x,t):\mathbb{R}^d\times [0,1] \rightarrow \mathbb{R}$ and the constructed sequence $\{ p_0,p_1,\cdots,p_T \}$, then the sequence is discretization of the time-evolving function $p(x,t)$ such that $p_0=p(x,0)=p^*$ and $p_T=p(x,1)=q$. If the sequence is sampled uniformly, then
$$
p(x,k \Delta t) = p_k(x), \Delta t = \frac{1}{T}
$$
where $T$ controls the fineness of the discretization.

We want the terminal variance $\sigma_q$ of the final distribution $p_T$ to be independent of $T$, the incremental variance can be defined as
$$
\sigma = \sigma_q \sqrt{\Delta t}
$$
as $x_t = x_{t-1} + \mathcal{N}(0, \sigma^2) = x_0 + \mathcal{N}(0,t \sigma^2)$ and $x_T = x_0 + \mathcal{N}(0, T \sigma^2) = x_0 + \mathcal{N}(0, \sigma_q^2)$.

Following description of DDPM, DDIM and Flow-matching will use $t\in [0,1]$ so the diffusion process can be written as
$$
x_{t+\Delta t} = x_{t} + \eta_t, \eta_t \sim \mathcal{N}(0, \Delta t\sigma_q^2)
$$
or
$$
x_t \sim \mathcal{N}(x_0, \sigma_t^2), \sigma_t^2 = t \sigma_q^2.
$$

Now, let's start the journey of DDPM, DDIM, and Flow-matching 😃.
