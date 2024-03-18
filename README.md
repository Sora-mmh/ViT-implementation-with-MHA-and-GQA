# ViT-implementation-with-MHA-and-GCA


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#About">About</a>
    </li>
    <li>
      <a href="#Module">Module</a>
    </li>
    <li>
    <a href="#MultiHeadAttention">Multihead Attention</a>
    </li>
    <li>
    <a href="#GroupedQueryAttention">Grouped-Query Attention</a></li>
  </ol>
</details>



<!-- ABOUT -->
## About
The [article](https://arxiv.org/pdf/2305.13245.pdf) of **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** endorsed by different LLMs since its release such as **Mistral Large**

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- Module -->
## Module
The architecture of this work is as follows:

 * [attention](ViT-implementation-with-MHA-and-GQA/attention) : contains two attention mechanisms explained later.
    * [__init__.py](ViT-implementation-with-MHA-and-GQA/attention/__init__.py)
    * [_gqa.py](ViT-implementation-with-MHA-and-GQA/attention/_gqa.py)
    * [_mha.py](ViT-implementation-with-MHA-and-GQA/attention/_mha.py)

 * [patches_embedder](ViT-implementation-with-MHA-and-GQA/patches_embedder) : divide the input images into patches through a 2D convolution, in which the kernel size is the patch size. Then embed each patch.
    * [__init__.py](ViT-implementation-with-MHA-and-GQA/patches_embedder/__init__.py)
    * [_base.py](ViT-implementation-with-MHA-and-GQA/patches_embedder/_base.py) 
  
 * [ViT](ViT-implementation-with-MHA-and-GQA/ViT) : assemble the different components.
    * [__init__.py](ViT-implementation-with-MHA-and-GQA/ViT/__init__.py)
    * [_base.py](ViT-implementation-with-MHA-and-GQA/ViT/_base.py)
  
 * [README.md](ViT-implementation-with-MHA-and-GQA/README.md)

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- MultiHeadAttention -->
## Multihead Attention


<!-- GroupedQueryAttention -->
## Grouped-Query Attention





