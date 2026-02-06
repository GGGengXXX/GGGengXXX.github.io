---
title: reading DiT code
date: 2026-02-06 03:28:36
tags:
---

# Time Embedding

we try to enrich the information of a time scalar!

if we dont do so, the info that the model can get from the time scalar is poor.

```python
def forward(self, t):
        # t_freq [batch, ..., t, dim]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        # t_emb [batch, ..., t, hidden]
        return t_emb
```

这里的 `t` 其实就是 `[batch, ]`

like `[4,4,4,4,4,4,...]` 

`timestep_embedding` using sin and cos to encode the number

code is like:

```python
@staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # freqs = exp(-log(max_period) * [0, ..., half - 1] / half)
        # [dim // 2]
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # t 应该是位置信息 [batch, ..., t, 1]
        # freqs [1, dim // 2]
        # args -> [batch, ..., t, dim // 2]
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # embedding -> [batch, ..., t, dim]
        return embedding
```

after the `timestep_embedding` we got the torch like [batch, dim]

after the mlp, we got [batch, hidden]

that is, TimestepEmbedder change from  `[batch]` to.`[batch, hidden]`



# LabelEmbedder

we are using **Classifier-Free Guidance (CFG)** 

先考虑条件生成，模型有时候的生成结果和条件拟合的不够好。比如生成的图片结果和你的 `text prompt` 在语义上靠的还不够近。这时候，你希望生成的结果再靠近条件一点，和条件更加贴合！

对于带条件的生成，我们往往需要先训练一个 Classifier，它的作用是对于给定的图像，输出它的分类类别。训练好了以后，他就有了看图的能力。

在扩散的过程中，每次 sample 一个噪音，使用Classifier对 input 求梯度，用这个梯度去 modify sample出来的噪音，给这个noise以引导。然后用这个noise去更新 `x` 

使用CFG就不用一个额外的classfier了。训练的时候，他会随机的丢弃标签，进行无标签的生成

形式化来讲，每次的输入是一个
$$
<x_t,t,c>
$$
**训练时：**

我们有概率地（`p=0.1~0.2`）令
$$
c = \empty
$$
 这样模型同时学会了无条件生成和带条件生成

**推理时**

走两次

1. $$
   input = <x_t,t, c>
   $$

2. $$
   input=<x_t,t,\empty>
   $$

然后两个结果进行综合
$$
\hat{\epsilon}=s\epsilon_{conditioned}+(1-s)\epsilon_{unconditioned}
$$
一般
$$
s = [5, 7]
$$
相当于有了一个 $\epsilon_{uncond}$ 又有了一个 $\epsilon_{cond}$  两者做差你就知道 $cond$ 的方向在哪里了！

`LabelEmbedder` 的 `__init__`

```python
def __init__(self, num_classes, hidden_size, dropout_prob):
```

这里 `dropout_prob` 应该是随机丢掉分类
