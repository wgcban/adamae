# *Ada*MAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders

- We propose *Ada*MAE, a novel, adaptive, and end-to-end trainable token sampling strategy for MAEs that takes into account the spatiotemporal properties of all input tokens to sample fewer but informative tokens.

- We empirically show that \adamae samples more tokens from high spatiotemporal information regions of the input, resulting in learning meaningful representations for downstream tasks.

![intro-fig](figs/adamae-intro-fig.jpeg)

## A comparision

Comparison of our adaptive masking with existing random *patch*, *tube*, and *frame* masking for masking ratio of 80\%.} Our adaptive masking approach selects more tokens from the regions with high spatiotemporal information while a small number of tokens from the background.

![mask-type-comp](figs/adamae-mask-types.jpeg)
