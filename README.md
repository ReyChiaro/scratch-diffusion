# scratch-diffusion
A repository for understanding, implementing denoise-diffusion/flow-matching models.

## References

[Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929v2)


## Overview of Diffusion

> The goal of generative models is constructing a **sampler** for an unknown distribution $p^{\star}(x)$. As long as the sampler is contructed, then we can generate **new** samples from the distribution.
> As the distribution is unknown, we should get a set of representative samples from the unknown distribution to estimate it.

We can directly learn a transformation between the unknown distribution and a simple-to-sample distribution such as Gaussian distribution. However this may be untractable as the target unknown distribution can be very large and complicated.
Diffusion models provide a general framework for learning such transformations, reducing the problem of sampling from distribution $p^{\star}(x)$ to a sequence of easier sampling problems.


### Gaussian Diffusion

Let random variable $x_0 \sim p^{\star} \in \mathbb{R}^d$, then construct a sequence of random variables $\{ x_1, x_2, \cdots, x_T \}$ by successively adding independent Gaussian noise with some small scale $\sigma$
$$
\begin{equation}
x_{t+1} = x_t + \eta_t, \eta_t \sim \mathcal{N}(0, \sigma^2)
\end{equation}
$$
which is called the *forward process*. Let $\{p_t\}_{t \in \{0,1,\cdots,T\}}$ be the marginal distribution of each $x_t$. We have
$$
\lim_{T\rightarrow \infty} p_T = \mathcal{N}(0, \sigma_q^2).
$$
In practice, $T$ is sufficiently large and we can assume that $p_T \approx \mathcal{N}(0, \sigma_q^2)$, where $\sigma_q$ is the standard deviation. What we *want to* do is to learn a transformation such that given the marginal distribution $p_t$, we can produce $p_{t-1}$. And if the final marginal distribution $p_T$ is given, we can produce $p_0=p^{\star}$ iteratively. This process is called *reverse process*, and the method to implemente this process is called *reverse sampler*.

The reverse process can be represented by conditional probability. At time step $t$, given the input $z$ sampled from $p_t$, the output of the reverse sampler generate a sample from the conditional distribution
$$
\begin{equation}
p(x_{t-1}|x_t=z).
\end{equation}
$$

However, though this implies that a generative model should learn the conditional distribution for every $x_t$ which counld be complicated. But we have the following insight, which will be proved in <u>DDPM sampler</u>:

> Fact 1 (Diffusion Reverse Process). For small $\sigma$ and the Gaussian diffusion process $x_{t+1} = x_t + \eta_t$, where $\eta_t \sim \mathcal{N}(0,\sigma^2)$, then the conditional distribution $p(x_{t-1} | x_t)$ is itself close to Gaussian.
> For all time step $t$ and condition $z \in \mathbb{R}^d$, there exists some mean parameter $\mu_{t-1} \in \mathbb{R}^d$ such that $p(x_{t-1} | x_t) \approx \mathcal{N}(x_{t-1}; \mu_{t-1}, \sigma_{t-1}^2)$.
