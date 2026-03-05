---
title: 'Paper Reading: A Survey On VLA Models for Embodied AI'
date: 2026-03-04 10:22:25
tags: paper reading
---

---

定义：embodied AI 领域的一类多模态模型 => (models：Vision Language Action)

V 模态

和 VLMs 相似，使用一些 vision foundation model 来作为 vision encoder ，从而得到 对于当前 environment 的视觉表征 (pretrained visual representation) 

L 模态

对 text 的instruction进行处理，使用LLM，得到 token embeddings

对视觉的 embeddings + 语言的 embeddings 做 alignment

**BLIP-2**（2023.1 ICML 2023）

冻结视觉模型和大语言模型，设置一个 **轻量** 的 **可学习的** Q-Former进行桥接

> Q-Former
>
> 输入是 learnable Query
>
> 先做 self attention  Q = K = V = learnable Query
>
> 接下来做 cross-attention
>
> Q = learnable Query, K = V = 视觉编码器的输出
>
> 数据输入是 高维度的视觉表征 + 32 维的 learnable query 输出是 32 维的视觉表征
>
> 实际上是对视觉的表征做了压缩
>
> 训练分为两个阶段
>
> 第一阶段进行 文本 和 图像特征的对齐， 模型学会了在图像中提取出语言的特征
>
> 第二阶段把提取的特征给语言模型，生成文字

再在机器人数据上微调，LLM可以预测 action

目前关于VLA的研究有三个方向

#### Components  组件研究

| 组件       | 作用                 |
| :--------- | :------------------- |
| 视觉表示   | 教机器人"怎么看"     |
| 动力学学习 | 教机器人"动作会怎样" |
| 世界模型   | 教机器人"预测未来"   |
| 推理能力   | 教机器人"思考        |

#### Low-level Control 底层控制

输入视觉和语言，决定怎么运动，输出往往是平移 + 旋转

#### High-level Planner 高层规划

把一个任务进行拆解再执行

## VLA Models

### components

#### RL

**DQN 2013 / 2015 **

输入像素图片 => 输出 action

本质是学习一个 $Q(s,a)$  的函数

把 4张图片 堆叠在一起 => 有运动的趋势

模型架构 CNN + 全连接 => 输出某个状态下，每个动作对应的 $Q$ 值

**DT(Decision Transformer)** 

序列预测，直接输出动作的序列，不预测Q值了

**RL + LLM**

由LLM来设置奖励函数

工作流程

```
1️⃣ LLM 写初始奖励函数
   ↓
2️⃣ 机器人用这个奖励函数训练
   ↓
3️⃣ 看训练效果怎么样
   ↓
4️⃣ 告诉 LLM"哪里不好
   ↓
5️⃣ LLM 修改奖励函数
   ↓
6️⃣ 重复 2-5，直到效果最好

```

#### PVR pretrained vision representation

视觉表征是对环境的观测 很重要！ 

**CLIP**

对文本 - 图片 做 embeddings

**R3M**

两个核心目标： 时间对比学习 + 视频语言对齐

**Time Constrastive Learning**

理解视频中多个视频帧、多个动作之间的先后顺序

会抽取三个帧，锚点帧，正样本（和锚点帧接近），负样本（和锚点帧很远）

一个视频可以采样得到很多这样的数据

**Video Language Alignment**

使用弱监督，一个视频对应的文本描述比较模糊，信息量较少。训练模型对video的表征中包含language的信息。

对于一个视频中的所有帧，他们都是使用同一个文本描述

损失中加入正则化，保证表征的稀疏性

> 稀疏最直观的表现就是输出的向量中有很多 0 。如果信息比较丰富的话，这个向量几乎都是非零值，那就会导致输出的表征中包含了太多杂乱的信息，Action模型在接受这个表征的时候，决策可能会被一些杂讯干扰，无法专注于动作本身

Loss 包含三个部分， 时间对比的损失 + 视频语言对齐的损失 + 正则化损失

**VIP**

> 什么是 `MAE`
>
> Mask AutoEncoder,
>
> AutoEncoder 由三个部分来组成 encoder bottleneck decoder
>
> 使用ViT
>
> encoder 把图片变成一个 latent，接着bottleneck强迫丢掉latent中的噪音，最后decoder 努力还原一开始输入的图片
>
> MAE 先把图片进行切片，分成一个个的patch，比如 $16\times 16$ 然后随机删除 75 % 的块，是模型还原原本的图片
>
> 这是一种自监督的学习，可以从互联网上获得无数的图片，使得模型学会 `完形填空`

**Voltron**



