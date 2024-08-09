# ViT-implementation-with-MHA-and-GQA

<!-- ABOUT -->
## About
The [article](https://arxiv.org/pdf/2305.13245.pdf) of **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** ,endorsed by different LLMs since its release such as **Mistral Large**, presents a new approach of allocating query heads with key and value heads when computing the scaled dot product.
This work englobes both standard multi-head attention, in which each query head is attributed to one key head, and grouped-query attention which allocates a subgroup of query heads to one key head.

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- Module -->
## Module
The architecture of this work is as follows:

 * [attention](/attention) : contains two attention mechanisms explained later.
    * [__init__.py](/attention/__init__.py)
    * [_gqa.py](/attention/_gqa.py)
    * [_mha.py](/attention/_mha.py)

 * [patches_embedder](/patches_embedder) : divides the input images into patches through a 2D convolution, in which the kernel size is the patch size. Then embed each patch.
    * [__init__.py](/patches_embedder/__init__.py)
    * [_base.py](/patches_embedder/_base.py) 
  
 * [ViT](/ViT) : assembles the different components.
    * [__init__.py](/ViT/__init__.py)
    * [_base.py](/ViT/_base.py)
  
 * [README.md](/README.md)

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->






