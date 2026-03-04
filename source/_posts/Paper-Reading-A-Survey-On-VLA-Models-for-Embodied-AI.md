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
4️⃣ 告诉 LLM"哪里不好"
   ↓
5️⃣ LLM 修改奖励函数
   ↓
6️⃣ 重复 2-5，直到效果最好

```

