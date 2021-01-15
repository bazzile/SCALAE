# SCALAE

Training and evaluation code for our paper: **Formatting the Landscape: Spatial conditional GAN
for varying population in satellite imagery**

We make additions to the existing ALAE architecture (based on StyleGAN), creating a spatially conditional version: SCALAE. This method allows us to explicitly disentangle a 2D input map from the model's latent vector and thus use the 2D input to guide the image generation. We use this method to generate satellite imagery from custom 2D population maps. 

[Arxiv link](https://arxiv.org/abs/2101.05069)

[Interactive colab](https://tinyurl.com/y2xa92t4)

We are very thankful to the original [ALAE repository](https://github.com/podgorskiy/ALAE) for the preprocessing and training code. For training setup please follow the instructions in the [original readme](https://github.com/LendelTheGreat/SCALAE/blob/master/README_ALAE.md)




