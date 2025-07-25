## DDPM Guidance



### SDE Perspective

$$
\hat{x}_{t-\Delta t} \leftarrow \mathbb{E}[x_{t-\Delta t} | x_t] + \mathcal{N}(0, \Delta t \sigma^2_q)
$$