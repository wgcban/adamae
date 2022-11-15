# *Ada*MAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders

*Left to right*: **(1)** Video, **(2)** Predicted video (mask ratio = 95%), **(3)** Prediction error (MSE), **(4)** Predicted categorical distribution, **(5)** Sampled mask

From $SSv2$ (samples from $50th$ epoch)

<p float="left">
  <img src="figs/ssv2-mask-vis-1.gif" width="400" />
  <img src="figs/ssv2-mask-vis-1.gif" width="400" /> 
</p>

![mask-vis-ssv2-1](figs/ssv2-mask-vis-1.gif) 
![mask-vis-ssv2-2](figs/ssv2-mask-vis-6.gif)
![mask-vis-ssv2-3](figs/ssv2-mask-vis-7.gif)
![mask-vis-ssv2-3](figs/ssv2-mask-vis-8.gif)
![mask-vis-ssv2-3](figs/ssv2-mask-vis-9.gif)
![mask-vis-ssv2-3](figs/ssv2-mask-vis-10.gif)
![mask-vis-ssv2-3](figs/ssv2-mask-vis-12.gif)
![mask-vis-ssv2-3](figs/ssv2-mask-vis-13.gif)

From $K400$ (samples from $50th$ epoch)

![mask-vis-k400-1](figs/k400-mask-vis-1.gif)
![mask-vis-k400-2](figs/k400-mask-vis-2.gif)
![mask-vis-k400-3](figs/k400-mask-vis-3.gif)
![mask-vis-k400-3](figs/k400-mask-vis-4.gif)
![mask-vis-k400-4](figs/k400-mask-vis-5.gif)
![mask-vis-k400-5](figs/k400-mask-vis-6.gif)
![mask-vis-k400-6](figs/k400-mask-vis-7.gif)
![mask-vis-k400-7](figs/k400-mask-vis-8.gif)


- We propose *Ada*MAE, a novel, adaptive, and end-to-end trainable token sampling strategy for MAEs that takes into account the spatiotemporal properties of all input tokens to sample fewer but informative tokens.

- We empirically show that *Ada*MAE samples more tokens from high spatiotemporal information regions of the input, resulting in learning meaningful representations for downstream tasks.

- We demonstrate the efficiency of *Ada*MAE in terms of performance and GPU memory against random *patch*, *tube*, and *frame* sampling by conducting a thorough ablation study on the SSv2 dataset.

- We show that our *Ada*MAE outperforms state-of-the-art (SOTA) by $0.7\%$ and $1.1\%$ (in top-1) improvements on $SSv2$ and $Kinetics-400$, respectively.

![mask-vis-1](figs/adamae-intro-fig.jpeg)


## A comparision

Comparison of our adaptive masking with existing random *patch*, *tube*, and *frame* masking for masking ratio of 80\%.} Our adaptive masking approach selects more tokens from the regions with high spatiotemporal information while a small number of tokens from the background.

![mask-type-comp](figs/adamae-mask-types.jpeg)

## Ablation experiments on SSv2 dataset

We use ViT-Base as the backbone for all experiments. MHA $(D=2, d=384)$ denotes our adaptive token sampling network with a depth of two and embedding dimension of $384$.  All pre-trained models are evaluated based on the evaluation protocol described in Sec. 4. The default choice of our *Ada*MAE is highlighted in gray color. The GPU memory consumption is reported for a batch size of 16 on a single GPU.

![ssv2-ablations](figs/adamae-ablations.png)


## Acknowledgement
Our AdaMAE codebase is based on the implementation of VideoMAE paper. We thank the authors of [**VideoMAE**](https://github.com/MCG-NJU/VideoMAE.git) for making their code available to the public.


## Citation
```
```
