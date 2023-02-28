# MDGCN
## Description
This is the repository for the PR paper [Multi-Level Graph Learning Network for Hyperspectral Image Classification].

Abstract: Graph Convolutional Network (GCN) has emerged as a new technique for hyperspectral image (HSI) classification. However, in current GCN-based methods, the graphs are usually constructed with manual effort and thus is separate from the classification task, which could limit the representation power of GCN. Moreover, the employed graphs often fail to encode the global contextual information in HSI. Hence, we propose a Multi-level Graph Learning Network (MGLN) for HSI classification, where the graph structural information at both local and global levels can be learned in an end-to-end fashion. First, MGLN employs attention mechanism to adaptively characterize the spatial relevance among image regions. Then localized feature representations can be produced and further used to encode the global contextual information. Finally, prediction can be acquired with the help of both local and global contextual information. Experiments on three real-world hyperspectral datasets reveal the superiority of our MGLN when compared with the state-of-the-art methods.


## Requirements

- Tensorflow (1.4.0)

## Usage

You can conduct classification experiments on hyperspectral images (e.g., Indian Pines) by running the 'trainGCN.py' file.

## Cite
Please cite our paper if you use this code in your own work:

```
@article{WAN2022108705,
title = {Multi-level graph learning network for hyperspectral image classification},
journal = {Pattern Recognition},
volume = {129},
pages = {108705},
year = {2022},
author = {Sheng Wan and Shirui Pan and Shengwei Zhong and Jie Yang and Jian Yang and Yibing Zhan and Chen Gong}
}
```
