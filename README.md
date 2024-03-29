# [CVPR'23] *Ada*MAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders

:book: Paper: [`CVPR'23`](https://openaccess.thecvf.com/content/CVPR2023/papers/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.pdf) and [``arXiv``](https://arxiv.org/abs/2211.09120v1)

Our paper (AdaMAE) has been accepted for presentation at CVPR'23.

### :bulb: Contributions:
- We propose *Ada*MAE, a novel, adaptive, and end-to-end trainable token sampling strategy for MAEs that takes into account the spatiotemporal properties of all input tokens to sample fewer but informative tokens.

- We empirically show that *Ada*MAE samples more tokens from high spatiotemporal information regions of the input, resulting in learning meaningful representations for downstream tasks.

- We demonstrate the efficiency of *Ada*MAE in terms of performance and GPU memory against random *patch*, *tube*, and *frame* sampling by conducting a thorough ablation study on the SSv2 dataset.

- We show that our *Ada*MAE outperforms state-of-the-art (SOTA) by $0.7\%$ and $1.1\%$ (in top-1) improvements on $SSv2$ and $Kinetics-400$, respectively.

### Method
![mask-vis-1](figs/adamae-intro-fig.jpeg)


### Adaptive mask visualizations from $SSv2$ (samples from $50th$ epoch)

| &nbsp; Video &nbsp;  | Pred. &nbsp;| &nbsp; Error &nbsp; | &nbsp; &nbsp; CAT &nbsp; | Mask | &nbsp; |  Video  | Pred. &nbsp;| &nbsp; Error &nbsp; | &nbsp; &nbsp; CAT  &nbsp; | Mask &nbsp; |
| ----------- | --------- | --------- | --------- | --------- |--|--------- | --------- | --------- | --------- | --------- |

<p float="left">
  <img src="figs/ssv2-mask-vis-1.gif" width="410" />
  <img src="figs/ssv2-mask-vis-2.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/ssv2-mask-vis-3.gif" width="410" />
  <img src="figs/ssv2-mask-vis-4.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/ssv2-mask-vis-5.gif" width="410" />
  <img src="figs/ssv2-mask-vis-6.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/ssv2-mask-vis-7.gif" width="410" />
  <img src="figs/ssv2-mask-vis-8.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/ssv2-mask-vis-9.gif" width="410" />
  <img src="figs/ssv2-mask-vis-10.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/ssv2-mask-vis-11.gif" width="410" />
  <img src="figs/ssv2-mask-vis-12.gif" width="410" /> 
</p>

### Adaptive mask visualizations from $K400$ (samples from $50th$ epoch):

| &nbsp; Video &nbsp;  | Pred. &nbsp;| &nbsp; Error &nbsp; | &nbsp; &nbsp; CAT &nbsp; | Mask | &nbsp; |  Video  | Pred. &nbsp;| &nbsp; Error &nbsp; | &nbsp; &nbsp; CAT  &nbsp; | Mask &nbsp; |
| ----------- | --------- | --------- | --------- | --------- |--|--------- | --------- | --------- | --------- | --------- |

<p float="left">
  <img src="figs/k400-mask-vis-1.gif" width="410" />
  <img src="figs/k400-mask-vis-2.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/k400-mask-vis-3.gif" width="410" />
  <img src="figs/k400-mask-vis-4.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/k400-mask-vis-5.gif" width="410" />
  <img src="figs/k400-mask-vis-6.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/k400-mask-vis-7.gif" width="410" />
  <img src="figs/k400-mask-vis-8.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/k400-mask-vis-9.gif" width="410" />
  <img src="figs/k400-mask-vis-10.gif" width="410" /> 
</p>
<p float="left">
  <img src="figs/k400-mask-vis-11.gif" width="410" />
  <img src="figs/k400-mask-vis-12.gif" width="410" /> 
</p>

### A comparision

Comparison of our adaptive masking with existing random *patch*, *tube*, and *frame* masking for masking ratio of 80\%.} Our adaptive masking approach selects more tokens from the regions with high spatiotemporal information while a small number of tokens from the background.

![mask-type-comp](figs/adamae-mask-types.jpeg)

## Ablation experiments on SSv2 dataset:

We use ViT-Base as the backbone for all experiments. MHA $(D=2, d=384)$ denotes our adaptive token sampling network with a depth of two and embedding dimension of $384$.  All pre-trained models are evaluated based on the evaluation protocol described in Sec. 4. The default choice of our *Ada*MAE is highlighted in gray color. The GPU memory consumption is reported for a batch size of 16 on a single GPU.

![ssv2-ablations](figs/adamae-ablations.png)

# Pre-training *Ada*MAE & fine-tuning:

- We closely follow the [VideoMAE](https://github.com/MCG-NJU/VideoMAE.git) pre-trainig receipy, but now with our *adaptive masking* instead of *tube masking*. To pre-train *Ada*MAE, please follow the steps in [``DATASET.md``](readme/DATASET.md), [``PRETRAIN.md``](readme/PRETRAIN.md).

- To check the performance of pre-trained *Ada*MAE please follow the steps in [``DATASET.md``](readme/DATASET.md) and [``FINETUNE.md``](readme/FINETUNE.md).

- To setup the conda environment, please refer [``FINETUNE.md``](readme/INSTALL.md).

# Pre-trained model weights

- Download the pre-trained model weights for SSv2 and K400 datasets [``here``](https://github.com/wgcban/adamae/releases/tag/v1).

## Acknowledgement:
Our AdaMAE codebase is based on the implementation of VideoMAE paper. We thank the authors of the [VideoMAE](https://github.com/MCG-NJU/VideoMAE.git) for making their code available to the public.



## Citation:
```
@InProceedings{Bandara_2023_CVPR,
    author    = {Bandara, Wele Gedara Chaminda and Patel, Naman and Gholami, Ali and Nikkhah, Mehdi and Agrawal, Motilal and Patel, Vishal M.},
    title     = {AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning With Masked Autoencoders},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14507-14517}
}
```
