# VariationalColdDiffusion

We present Variational Cold Diffusion (VCD), a framework for diffusion-like systems that couple a deterministic degradation opera- tor and a variational encoder to guide the reconstruction process. VCD separates the entropy of the forward and the reverse process, allowing for full generative control via the variational encoder. We demonstrate the benefits of this approach by exploiting the inductive bias of blurring degradation, which gives rise to an image generation process that iteratively adds low- to high-frequency image features. We find that data- level perturbation during training is essential to sample deterministic diffusion models sequentially, and propose two methods to achieve this: incorporating the model’s own bias into training and using minimal noise injections for perturbation. Our work contributes to understanding the dynamics in (deterministic) diffusion models and the role of Gaussian perturbation therein, and provides a general framework for controlling the inductive biases of arbitrary deterministic degradation operators.



![TransYNet architecture](TransYNet_architecture.png "TransYNet architecture")
