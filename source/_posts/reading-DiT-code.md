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

这里 `dropout_prob` 应该是随机丢掉分类的概率

```python
use_cfg_embedding = dropout_prob > 0
self.embedding_table = nn.Embedding(num_classes  + use_cfg_embedding, hidden_size)
```

如果不打算使用 CFG 的话，dropout_prob 可以设置为一个负数

如果设置为正数，表示 `dropout` 的概率

那么在注册 `Embedding_table` 的时候会多加一个分类

`forward` 的定义

```python
def forward(self, labels, train, force_drop_ids=None):
```

`train` 是一个 `boolean` 决定是否在训练模式

`labels` 是一个标签的列表 `[batch, ]`

```python
def forward(self, labels, train, force_drop_ids=None):
        # 无分类引导
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
          # token_drop 随机把一些标签打掉
            labels = self.token_drop(labels, force_drop_ids)
        # 然后把labels拿去做embeddings
        embeddings = self.embedding_table(labels)
        return embeddings
```

总的来说，`Label_embedder` 

do such changes

`[batchsize, ]` -> `[batchsize, hidden_size]`

支持随机的把一些标签置空

```python
labels = torch.where(drop_ids, self.num_classes, labels)
```

如果 `drop_ids` 对应的位置是 `True` 就把这里的 `label_id` 值换成 `self.num_classes` 表示 `invalid` 

## DIT Block

`forward` 实现

```python
def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

先看 `adaLN_modulation` 的实现

```python
self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
```

输入是条件 `c` 

这里的 `c` 是把 `time_embedding` 和 `label_embedding` 加在一起

在 `class DiT` 中使用了 `DiTBlock`

```python
self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
```

为什么使用 `nn.ModuleList` 而不是直接使用 `[]` 呢，使用 `nn.ModuleList` `pytorch` 才能看见这些模型，会为模型自动注册对应的参数。和 `nn.Sequential` 的区别在于，`nn.Sequential` 中数据会自动流过每一 `layer` 而 `nn.ModuleList` 更像是一个容器而已

节选自 `DiT` 的 `forward`

```python
x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
```

可以看到条件就是把 time embedder, label embedder 的结果加在一起



Forward 中的 `adaLN_modulation` 返回一个长度为 6 的列表，每一个元素就是 `[batch_size, hidden_size]`

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
```

一个 `DitBlock` 分为两个阶段：

- **MHSA（多头自注意力）\**负责\**“横向沟通”**：它的作用是让不同的 Patch（Token）互相看一看，理解空间关系（比如：这个补丁里的猫耳朵和另一个补丁里的猫眼睛是什么关系）。
- **FFN（前馈网络）\**负责\**“纵向挖掘”**：在注意力机制帮 Token 收集完周围的信息后，FFN 负责对这些收集到的信息进行深度处理和非线性变换，将原始特征转化为更高级的概念表示。

代码实现

```python
x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
```

其中 `modulate` 实现，残差连接 + （乘以 scale + shift）

```python

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```



其中 `mlp` 就是 `FFN` 前馈网络(Pointwise Feedforward)

多头注意力实现

```python
self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
```

前馈网络，`DitBlock` 中的定义

```python
self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
```

实际上 `Mlp` 中会经历一个先提升维度，再降低维度的过程

以下是 `Mlp` 的具体实现

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer):
        super().__init__()
        # 第一层：将维度从 in_features (hidden_size) 增加到 hidden_features (通常是 4倍)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # 第二层：将维度从 hidden_features 还原回 in_features
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0)

    def forward(self, x):
        x = self.fc1(x)  # 升维操作
        x = self.act(x)  # 非线性激活
        x = self.fc2(x)  # 降维操作
        return x

```

各个模块搭建好了，接下来可以看 `DiT` 了

输入是 `x`  使用 `VAE` 进行下采样

`256, 256` -> `32, 32` 论文中 `VAE` 后的通道数为 `4`

输入 `x` `batch_size, 4, 32, 32` 

```python
x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
```

