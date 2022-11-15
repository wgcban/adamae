# *Ada*MAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders

- We propose *Ada*MAE, a novel, adaptive, and end-to-end trainable token sampling strategy for MAEs that takes into account the spatiotemporal properties of all input tokens to sample fewer but informative tokens.

- We empirically show that \adamae samples more tokens from high spatiotemporal information regions of the input, resulting in learning meaningful representations for downstream tasks.

![intro-fig](figs/adamae-intro-fig.jpeg)

## A comparision

Comparison of our adaptive masking with existing random *patch*, *tube*, and *frame* masking for masking ratio of 80\%.} Our adaptive masking approach selects more tokens from the regions with high spatiotemporal information while a small number of tokens from the background.

![mask-type-comp](figs/adamae-mask-types.jpeg)

## Ablation experiments on SSv2 dataset

We use ViT-Base as the backbone for all experiments. MHA $(D=2, d=384)$ denotes our adaptive token sampling network with a depth of two and embedding dimension of $384$.  All pre-trained models are evaluated based on the evaluation protocol described in Sec. 4. The default choice of our *Ada*MAE is highlighted in gray color. The GPU memory consumption is reported for a batch size of 16 on a single GPU.

![ssv2-ablations](figs/adamae-ablations.png)

