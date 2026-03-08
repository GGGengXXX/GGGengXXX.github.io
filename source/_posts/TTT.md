---
title: TTT
date: 2026-03-08 16:11:20
tags: paper
---

---

Let's focus on Test Time Training

## Test-Time Training with Self-Supervision for Generalization under Distribution Shift

设计了一个 Y 形的网络结构：

首先是树干，这是一个 **共享特征提取器** 

然后有一个 **主任务分支** 一个 **辅助任务分支** 都接在共享特征提取器后面

主任务分支是一个图像分类的网络

辅助任务分支是一个自监督的网络

> 什么是自监督：
>
> 数据的标签来源于数据本身
>
> 这里的话是去吧图片旋转一定度数的到新样本，标签就是旋转的度数。这个标签 **不依赖于人工标注** 可以从数据本身获取
>
> 还有一个典型的自监督例子就是 GPT Bert 这些大语言模型，原本的文本数据挖个空来做完形填空，同样不需要人工标注数据

然后测试的时候，测试的样本是没有标签的，但是可以通过自监督产生标签啊！所以模型会在测试的时候进行一次自监督的训练，更新共享特征提取器，然后再使用更新后的 **共享特征提取器** 进行预测



为什么选择 **图片旋转** 这个自监督的任务：因为为了预测出图片旋转的度数，模型不得不去理解图片

## Learning to (Test-Time) Train: RNNs with Implicit Latent State

> 什么是 RNN
>
> 引入一个变量 $h$ 
>
> 对于每一个 $h_{t}$  它通过上一个 $h_{t-1}$ 输入 $x_t$ 
>
> 需要并行 梯度消失

所有的 序列建模层 Sequence Modeling Layers 可以被看做是在把历史上下文 [historic context] 存进隐藏层中

例如LLM 就是把海量的语料压缩存进模型的weights中

训练模型，更新模型参数就是一种把 训练数据集 压缩存进模型的方法

在LLM中，做了Next-Token-Prediction的自监督任务，实现了对语料的压缩

我们把context作为dataset，在ttt层做自监督学习：

自监督的任务是把这个输入的 $x_t$  乘以一个低秩的矩阵 $\theta_K$ 在论文中被称为 多视图重构 (multi-view reconstruction) 

也就是 corrupted input 然后要 还原得到完整的 input

这里的 $\theta$ 也是模型的参数，相当于 attention 当中的 $W_Q, W_K, W_V$ 在训练前随机初始化，训练的时候更新，推理的时候不动了

如果 TTT 只是做简单的重建的话，那么TTT是没有 `outer loop parameters` 的，比如 `self attention` 就会有 $W_Q,W_K,W_V$ 会在整个训练过程中更新。但是 TTT 此时还只是做重建

 

   
