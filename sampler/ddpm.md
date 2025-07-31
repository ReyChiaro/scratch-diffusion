## References

[Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929v2)


## Review

Diffusion modeling aims to construct a sampler of a target distribution $p^*$. The idea is not to learn the mapping between a easy-to-sample distribution $q$, but construct a reverse sampler on a sequence of distributions $\{ p_{k\Delta t} \}_{k=0}^{k=1/{\Delta t}}$.

- Sequence of distributions: these distributions interpolate between the easy-to-sample distribution and the target distribution such that $p_1=q, p_0=p^*$. (The symbol may be different from some papers)
- Sequence of distributions has 

## DDPM: Stochastic Sampling

  