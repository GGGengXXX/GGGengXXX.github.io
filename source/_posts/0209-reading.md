---
title: 0209_reading
date: 2026-02-07 21:09:09
tags:
math: true
---

## Pre knowledge

光流和PointTracking ：光流指的是在相邻的两张图片 点 $(u_1,v_1)$ 到 点 $(u_2,v_2)$ 所组成的矢量，而pointTracking(2D)指的是更长时的，一个点在所有帧上的轨迹，而不只是关注相邻的两帧。

## 4D表征：

PointWorld：3D点流，分为状态和动作，状态使用场景RGB-D图，动作使用机器人URDF推算

Any4D：视频，深度图，雷达多普勒，相机位姿

Uni-Enter：voxel体素

Ego-Twin：视频 + 文本描述 + 骨骼4D

AnimateAnyMesh：关节连接矩阵 + 初始结构点云 + 移动轨迹

Dream2Flow：深度图到3D点云 (object flow)

MoCapAnything：骨骼结构、Mesh网格、Image图片；图片帧使用DinoV2编码、视频到Mesh重建

D4RT：连续的视频帧

对于每一篇文章，我们关心两个问题

- 如何表征 4D 的
  - 数据如何获取，如何得到输入
  - 如何进行编码，模型的具体架构？
- 对我们的工作有什么启发

接下来，我们将对每一篇文章展开介绍。

---

### Dream2Flow

这篇工作实现了对机器人要操作的物体的表征，输入是一个初始帧 $I_0$ 然后使用视频生成模型生成 $I_1, I_2, ...$ 

> 视频生成：生成目标物体运动的视频

然后用 `SpatialTrackerV2` 得到深度图，用 `Grounding DINO` 得到 **物体的检测框** 进一步用 `SAM2` 得到物体的mask，然后投到3D就得到物体的点云4D序列了。

得到这个4D序列是希望能预测4D的序列，接下来我们关注三种类型的任务，以及我们分别是怎么解决的。

#### Push-T

对于一个在水平面上的物体，给他一个push的力，这坨点云会怎么动？我们的输入是一个14维度的query，包括 3D 的位置，3D的颜色，3D的法向量（取右边和下面的像素点 $B$ 和 $C$，投到3D上，然后计算 

$$\vec{AB}\times \vec{AC} $$

 作用点 $(u,v)$ 和推的方向 $(\Delta u, \Delta v)$ 拖拽距离 $d$ 

接下来的问题在于模型架构，文中没有明确的说明，只说了基于`PointNet` 

猜测：每一个点的信息是各不相同的，但是推动作用点 $(u,v)$ 和推动的方向还有拖拽距离是都一样的，拖拽信息会给每一个点的 `14 dimension` query 都复制一份。然后经过一个 `mlp` ，再来一个全局池化综合一下信息，然后过一个 `mlp` 得到每一个点的位移预测信息。

#### Real-World Domain

任务是抓起来一个物体

做了一些假设：末端执行器（机器人的手）碰到物体就和物体融为一体了，刚体运动。

后续使用的是一个机器人的求解器，基于数学的方法。

---

### PointWorld

论文标题：`PointWorld Scaling 3D World Model for In-The-Wild Robotic Manipulation`

解读标题：

一个 **Scaling** 的3D世界模型，使用了大量的机器人数据，为复杂、多任务的机器人操作任务服务

同样也是用4D点流表征，和 `Dream2Flow` 的区别在于，`Dream2Flow` 好像没有使用什么模型架构，而是提出了一种新的范式，可以使用4D点流作为 `Reward` 实现机器无关的具身智能训练。`PointWorld` 是有模型架构的。

有模型就会有输入输出。

论文中将模型表述为
$$
S\times A\rightarrow S
$$
其中 `S` 表示的是场景的点云， `A` 表示的是 action space

和很多自回归的方法不同，一次性会推断出未来 `H` 步的场景

这里的输入还是4D点流。

对于静态的场景，使用静态点云来表示。使用Point Transformer v3来处理点云。不带点追踪，减轻了很多负担

> Point Transformer v3 （后面称为PTv3）是用来处理点云的 `Transformer` 注意PTv3 的输入是代表点云的语义信息，例如 $N$ 个点的点云，输入就应该是 $(N, hidden\_size)$  

对于机器人，是带时间序列 `T` 的，对于 $N_R$ 个机器人上的点，就可以出来 $T\times N_R\times3$ 的点云，还要进行一系列的特征化

> 注意这里的点是采样过的点，只选择了机器人的夹持器上的点， 因为只有这些点直接与场景进行交互

静态场景的点云是静态的，但是每一个静态的场景会和动态的每一帧进行拼接

得到 $T\times(N_R+N_S)\times hidden\_size$ 

这里的机器人点云和场景点云分别进行了不同的特征化：

- 机器人点云编码了位置信息、时间步、颜色标记、法线、速度和加速度

- 场景点云编码了位置信息、2D语义信息、外观特征

而这些可以统一的交给 `PTv3` 可以处理混合编码

---

接下来我们关注 4D 重建的部分论文

### Any4D

[search in alphaxiv](https://www.alphaxiv.org/abs/2512.10935) 25-12-10

这一篇的代码是开源的

论文标题：Any4D: Unified Feed-forward Metric 4D Reconstruction

模型实现了4D场景的重建

可以 fomulate 成下面的形式
$$
(\tilde{s},\{\tilde{R}_i, \tilde{D}_i, \tilde{T}_i,\tilde{F}_i\}^{N}_{i=1})=Any4D(I,O)
$$
这里的输入可以分为基础输入 $I$ 和额外输入 $O$ 

基础输入是RGB的视频流，$[N,H,W,3]$

此外还可以选择性地输入深度图 + 多普勒流场（速度场）+ 相机位姿（相机外参：分为平移 + 旋转） + Rays场（相机内参）

代码实现中：

Optional Input 使用了两种类型的 `encoder`

对于深度图，Rays场，场景流，使用 `ViTEncoderNonImageInput`

对于 相机位姿（旋转 + 平移）尺度因子使用 `EncoderGlobalRepInput`

论文的 `3.1` 介绍了模型的架构

> 模型的架构分为了三个部分：输入的 `encoder` 中间的 `transformer` 最后输出的对于每一个 view 的`decoder`

#### encoder

图片使用DinoV2 最终编码成 `[hidden_size=1024, H / 14, W / 14]` 

其他的模态分别使用 `CNN` 或者 `MLP` 编码成相同的维度

> 深度图：对于每一个 view 做独立的归一化，然后使用一个 `shallow CNN` 进行编码
>
> 多普勒(Doppler Velocity, 速度场)：对第一个view做归一化，然后往后的所有view沿用第一个view的归一化的参数，同样使用 `CNN based Encoder`
>
> 相机内参：相机的内参和rays是可以可逆转换的，这里使用rays（射线场）来表征，同样使用 `CNN` 把 3 channels map到 1024的hidden_size
>
> 相机位姿：分别使用两个 四层的MLP，把维度拉到 1024，使用全局（所有views）的归一化，同时加入一个表征当前是第几帧的position embedding
>
> metric scale token：深度图和相机位姿的归一化分别可以得到一个 `s` 然后使用两个 `MLP` 就可以得到 1024 的latent 了

然后得到若干独立的 `tokens`

把这些tokens直接相加，对于每一个 `view` 就可以得到 `[hidden_size=1024, H/14, W/14]`

把后两个维度展平，对于每一个 `view` 可以得到 `[M = H / 14 * W / 14, hidden_size = 1024]` 

也就是，对于每一个view，可以得到 `M` 个 `token` ，每个 `token` 维度为 `(1024, )` 

总共有 `N` 个 `view` ，就会有 $N\times M$ 个token

#### transformer

使用一个交替注意力transformer (alternating-attention transformer)

输入是 $N\times M + 1$ 个token，还有一个是一个 `learnable` 的参数，来表示 `scale`

#### output representation head

**Geometry DPT Head** 

预测每一个view 的深度图 $D$ 和 射线图 $R$ 以及一个 $confidence$ 表示置信度

> DPT: Dense Point Transformer: 使用transformer 来实现逐个像素的预测输出

**Motion DPT Head**

输出每一个点的场景流 $F_i$

- 以第一帧确定世界坐标系（第一帧图片的中心）
- 只track第一帧中出现的点

> 这里的输出是一个 `[N, 3, H, W]` 的格式
>
> 表示的是这个点相对于第一帧，在 $x,y,z$ 三个方向上发生的偏移

**Pose Decoder**

平均池化的 `CNN`

输出 up-to-scale 比例缩放的 相机位移和相机旋转 $T_i$

**Metric Scale Decoder**

一个 lightweight 的 MLP 用于预测scale系数的log值（原来的数字很大）

总结来说，我们得到了以下的输出

- scale缩放系数 $S$
- 相机的位姿 $T$ 相机外参
- 射线图 $R$ 相机内参

- 深度图 $D$
- 场景流 $F$ 

我们来试试完成4D重建的任务，这些条件是否充分(当然是充分的hhh)

完成一个场景的重建
$$
SceneRecon = D\times S\times (T\times R)
$$
场景中像素的真实运动
$$
Motion=S\times F
$$
动态4D预测
$$
SceneRecon'=SceneRecon + Motion
$$


---

### D4RT

---

### AnimateAnyMesh

这一篇的代码是开源的

对主体的表征不需要骨骼！

理论上只要有点云，有Mesh就能做～其中点云需要是带pointTracking的4D序列才行

输入包括三个部分

- 主体的Mesh网格

  - >  这里其实用Mesh来建立图的邻接关系，表征主体的拓扑结构，代替了骨骼的作用？

- 点云(initial 点云)

- 点云的运动序列

**overview：**

首先训练 `encoder` 和 `decoder` 实现将一个运动序列编码为 `latent` 在 decode 出来

这里没有使用 bone，需要的是mesh 网格；

什么是 `mesh` 就是在 pointCloud 的基础上，添加三角面片，有了这些三角面片实际上相当于把孤立的点云表示建模成了类似 graph 的结构

然后使用 DiT 去生成运动序列的 `latent` 然后再使用训练好的 `decoder` 就好了

需要用 text 去引导diffusion 的过程，这里使用的是 MMDiT，Rectified Flow

训练 `MMDiT` 生成从 noise 到生成目标的方向向量



#### DyMeshVAE

**encode**

```python
def encode(self, pc, faces=None, valid_mask=None, adj_matrix=None):
        # pc 维度: [B, T, N, 3] (输入动态序列)
        B, T, N, D = pc.shape
        device = pc.device
```

模型的输入如此，传入的点云形状是 `[B, T, N, D]` 

然后得到第一帧的点云

```python
        # 1. 提取初始帧
        pc0 = pc[:, 0]  # [B, N, 3]
```

接下来是得到每一帧想对于初始帧的相对位移

```python
# 2. 计算相对轨迹（差分）并展平时间维度
        # (pc - pc[:, :1]) -> [B, T, N, 3] (每一帧减去第一帧)
        # .permute(0, 2, 1, 3) -> [B, N, T, 3]
        # .flatten(2, 3) -> [B, N, T*3]
        pct_rel = (pc - pc[:, :1]).permute(0, 2, 1, 3).flatten(2, 3) # [B, N, T*3]
```

对初始点云和轨迹点云分别进行embed

```python
       # 3. 映射到特征空间 (Embedding)
        pc0_embed = self.point_embed(pc0)      # [B, N, C]
        pct_embed = self.traj_embed(pct_rel)   # [B, N, C]
```

`pc0_embed` 和 `pct_embed` 分别代表了主体的静态点云信息和轨迹信息

如果有 `adj_matrix` 的话，需要对 `pc0_embed` 做 `self-attention` 然后把 `adj_matrix` 作为 `mask`

> 需要让 `pc0_embed` 了解拓扑信息，比如手指上的关节的两个点距离和手指上的一个点和大腿上的一个点的距离都很近，但是手指上的关节的两个点是要一起运动的，而手指和大腿是不会一起运动的

```python
# 4. 拓扑信息聚合 (公式 2: 结合 Adj 矩阵的 Self-Attention)
        if adj_matrix is not None:
            # adj_matrix 维度: [B, N, N]
            for neighbor_layer in self.neighbor_layers:
                # 这里的 mask 确保只在相连顶点间算 Attention
                pc0_embed_res = neighbor_layer(pc0_embed, key=pc0_embed, value=pc0_embed, mask=adj_matrix) # [B, N, C]
                pc0_embed = pc0_embed + pc0_embed_res # [B, N, C]
```

过 `n` 遍 `self-attention` + 残差连接，修改结果保存在 `pc0_embed`

接下来要做最远点采样，我们先保存一版采样前的结果

```python
# 保存全量顶点特征，用于后续 Cross-Attention 的 Key 和 Value
        pc0_embed_ori = pc0_embed # [B, N, C]
        pct_embed_ori = pct_embed # [B, N, C]
```

然后应用最远点采样，得到采样结果 `idx` 然后更新 `pc0_embed` 和 `pct_embed` 

```python
# 5. 最远点采样 (FPS)
        with torch.no_grad():
            valid_length = valid_mask.sum(dim=-1)
            # 从 N 个点中选出 K 个代表点的索引
            _, idx = ops.sample_farthest_points(points=pc0_embed, lengths=valid_length, K=self.num_traj) 
            # idx 维度: [B, K]
            idx = replace_negative_indices(idx, valid_length)

        # 6. 根据索引提取代表点的特征 (Gather)
        # pc0_embed: [B, K, C]
        pc0_embed = torch.gather(pc0_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pc0_embed.shape[-1]))
        # pct_embed: [B, K, C]
        pct_embed = torch.gather(pct_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pct_embed.shape[-1]))

```

这时， `pc0_embed` 是采样后的版本，`pc0_embed_ori` 存储采样前的结果

接下来做 `cross_attention`

- Query = pc0_embed （采样后）
- Key = pc0_embed_ori (采样前)
- Value1 = pc0_embed_ori(采样前)
- Value2 = pct_embed_ori(采样前)

每次过完一个 `attention` 后接一个前馈网络

```python
# 7. 编码器块 (公式 3 & 4: Cross-Attention 聚合全局信息)
        for enc_attn, enc_ffn in self.enc_blocks:
            # q_stream: 代表点 [B, K, C]
            # k/v_stream: 原始全量点 [B, N, C]
            attn_res_0, attn_res_t = enc_attn(
                q_stream=pc0_embed, 
                k_stream=pc0_embed_ori, 
                v1_stream=pc0_embed_ori, 
                v2_stream=pct_embed_ori
            )
            pc0_embed = pc0_embed + attn_res_0 # [B, K, C]
            pct_embed = pct_embed + attn_res_t # [B, K, C]
            
            # FFN 层处理
            ffn_res_0, ffn_res_t = enc_ffn(pc0_embed, pct_embed) # [B, K, C]
            pc0_embed = pc0_embed + ffn_res_0 # [B, K, C]
            pct_embed = pct_embed + ffn_res_t # [B, K, C]
```

接下来我们需要一个latent来表征原始的静态结构，直接全连接就好了

这大概是一个重建任务吧，所以没有什么不确定性，目标是和原本的初始动作一样就好了

```python
# 8. VAE 潜在空间映射 (公式 5 & 6)
        # 形状特征 x0: 不做 KL 约束
        x0 = self.mean_fc_x0(pc0_embed) # [B, K, C_latent]
```

接下来是轨迹采样，需要有不确定性了

VAE采样过程：

- 得到 `mean` 和 `logvar`

```python
mean = self.mean_fc_xt(pct_embed)
logvar = self.logvar_fc_xt(pct_embed)
```

- 接下来是采样

> 直接采样高斯分布是不可导的，所以需要使用重参数化
>
> ```python
> # 无法求梯度
> z = torch.normal(mean=mean, std=std)
> # 先采样一个随机噪声（不参与梯度的计算）
> epsilon = torch.randn_like(mean)
> z = mean + torch.exp(0.5*logvar) * epsilon
> ```

```python
# 重参数化采样 (Reparameterization)
  posterior = DiagonalGaussianDistribution(mean, logvar)
  xt = posterior.sample() # [B, K, C_latent]
  kl = posterior.kl()     # [B, K] 或标量
```

最后把 $x_0$ 和 $x_t$ 拼接在一起

返回 `kl` encode的最终结果`x` 采样的结果 `idx` ，原始的未采样的 `pc0_embed_ori`  

**decode**

定义及传入参数

- encode 中拼接了初始的 $x_0$ 和 $x_t$ 得到这里输入的 `x`
- queries 是原始的 `N` 个点 $[B,N,3]$
- pc0_embed_ori 是特征增强的原始的 `N` 个点 `[B,N,C]`

```python
def decode(self, x, queries, pc0_embed_ori):
```

首先拆掉拼接在一起的 $x_0$ 和 $x_t$ 

然后把他们从 `C_latent` 维度投影到 `C` 维度

```python
# 1. 拆分形状 Latent 和 动作 Latent
  x0_latent, xt_latent = x.chunk(2, dim=-1) # 分别为 [B, K, C_latent]

  # 2. 投影回隐藏层维度
  x0 = self.proj_x0(x0_latent) # [B, K, C]
  xt = self.proj_xt(xt_latent) # [B, K, C]
```

然后是解码自注意力

```python
# 3. 解码器自注意力块 (代表点之间进行信息交换)
for dec_attn, dec_ffn in self.dec_blocks:
    # Self-Attention: 代表点之间互相观察，优化动作逻辑
    attn_res_0, attn_res_t = dec_attn(
        q_stream=x0, 
        k_stream=x0, 
        v1_stream=x0, 
        v2_stream=xt
    )
    x0 = x0 + attn_res_0 # [B, K, C]
    xt = xt + attn_res_t # [B, K, C]

    # FFN 层
    ffn_res_0, ffn_res_t = dec_ffn(x0, xt)
    x0 = x0 + ffn_res_0 # [B, K, C]
    xt = xt + ffn_res_t # [B, K, C]
```

把全量点的形状特征作为 `Query`

使用 `cross-attention` 

$x_0$ 作为 `Key` 

而轨迹的特征 $x_t$ 作为 `Value`

使得每一个点(全量)都能查询到关于运动轨迹的信息

```python
# 4. 最终交叉注意力 (从 K 个代表点上采样到 N 个原始点)
  # query_embed: 将全量点的形状特征作为查询 [B, N, C]
  query_embed = self.fc_query(pc0_embed_ori) # [B, N, C]

  # Cross-Attention: 
  # 每个原始点 (N) 去询问代表点 (K)：“我该怎么动？”
  # key=x0 (形状参考), value=xt (动作参考)
  latents = self.decoder_final_ca(query_embed, key=x0, value=xt) # [B, N, C]
```

这时，每个点都得到了关于运动轨迹的信息，就可以开始做输出了

其实就是在后面接一个 `Linear` (不是`mlp`)

```python
# 5. 投影到 3D 坐标偏移空间
# self.to_outputs 将 C 维映射到 (T-1)*3 维
outputs = self.to_outputs(latents) # [B, N, (T-1)*3]
```

然后就是调整一下维度 由于我们得到的都是相对于第一帧的位置偏移，所以`outputs` 中的每一个输出都要与第一帧的点云相加，才能得到4D的运动序列

```python
# 6. 重塑维度还原为序列格式
# .view(...) -> [B, N, T-1, 3]
# .permute(0, 2, 1, 3) -> [B, T-1, N, 3] (时间维度排在前面)
outputs = outputs.view(x.shape[0], queries.shape[1], -1, 3).permute(0, 2, 1, 3) # [B, T-1, N, 3]

# 7. 合成最终动画 (公式 1)
# queries[:, None] 维度是 [B, 1, N, 3] (初始帧)
# 将初始位置与每一帧的相对位移相加
outputs = queries[:, None] + outputs # [B, T-1, N, 3]

return outputs # 返回完整的动态序列 (不含第一帧，或根据实现包含第一帧)
```

#### Shape-Guided Text-to-Trajectory Model

基于 `MMDiT`  

> 传统的 `DiT` 对 `image` 做 self attention 然后对 `text` 做 `cross attention` 其中 image 是主体 text 是条件
>
> MMDiT: MultiModel DiT
>
> 把 `text` 和 `image` 同时做 `self-attention`
>
> 传统的 `DiT` 只有 `image` 到 `text` 的 query
>
> 而在MMDiT中，同时存在
>
> - image -> image
> - Text -> text
> - image -> text
> - Text -> image
>
> 的query

**整体结构**

```tex
输入: x (动作Latent噪声) [B,K,C], t (时间步) [B], texts (文本列表)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    timestep_embedding  CLIP Text       input_proj
    + time_embed(MLP)   Encoder         (Linear)
          │             + clip_token_mlp      │
          ▼               ▼               ▼
        t_emb          text_embed          h
       [B, W]          [B, 77, W]       [B, K, W]
          │               │               │
          └───────┬───────┘               │
                  ▼                       ▼
            ┌─────────────────────────────────┐
            │   Transformer_cogx (×N layers)  │
            │   CogXAttentionBlock:           │
            │     - AdaLN-Zero (x & text)     │
            │     - Joint Self-Attention       │
            │     - Joint MLP                  │
            └─────────────────────────────────┘
                          │
                    output_proj → 预测噪声/速度场 [B, K, C]
```

这一部分考虑了文本信息和静态结构，输出是一个速度场，用来为扩散过程提供指导。

每一步扩散都会调用这个 `MMDiT` 得到引导

#### **Diffusion Pipeline**

Training

得到时间步

```python
# ---- 1. 采样时间步 ----
    times = torch.rand(x_start.shape[0], device=x_start.device)  # [B] — t ~ U(0,1)
    padded_times = append_dims(times, x_start.ndim - 1)           # [B, 1, 1] — 扩展维度以广播
    
    # ---- 2. 构造加噪样本 ----
    t = cosmap(padded_times)                           # [B, 1, 1] — cosine 重映射后的时间步
    x_t = t * x_start + (1. - t) * noise              # [B, K, C] — 线性插值 (t=1→数据, t=0→噪声)
    
    # ---- 3. 保护 f0 通道: 用原始 x_start 的 f0 替换加噪版本 ----
    # x_start[:, :, :f0_channels] → [B, K, f0] 静态形状，不加噪
    # x_t[:, :, f0_channels:]     → [B, K, ft] 动态运动，已加噪
    x_t = torch.cat([x_start[:, :, :f0_channels], x_t[:, :, f0_channels:]], dim=-1)  # [B, K, C]
```

其中 `cosmap` 会做时间步的重映射，中间的时间步 例如 $t=0.5$ 会被采样得更多

$x_t$ 会得到 静态形状 + 动态运动的混合张量，只有动态运动的部分混上噪声

然后计算 `flow` 也就是从 `noise` 到 `x_start` 的直线方向

```python
# ---- 4. 计算目标 flow ----
    flow = x_start - noise     
```

然后使用 `DyMeshMMDiT` 考虑 `text` 、 运动轨迹，期望得到的是从 `noise` 到 `x_start` 或者预测结果的 方向向量

```python
# ---- 5. 模型前向传播 ----
# model = DyMeshMMDiT, 调用其 forward(x_t, t, texts=...)
model_output = model(                              # [B, K, C] — 模型预测的 flow 或 noise
    x_t,                                           # [B, K, C] — 加噪样本 (f0 未加噪)
    t.squeeze(-1).squeeze(-1),                     # [B]       — 时间步 (去掉扩展的维度)
    **model_kwargs
)
```

然后把 `model_output` 和实际得到的 `flow` 去做 `mse` 得到 `loss`

```python
# ---- 6. 选择预测目标 ----
if predict == 'flow':
    target = flow                                  # [B, K, C] — 目标: x_start - noise
elif predict == 'noise':
    target = noise                                 # [B, K, C] — 目标: 噪声本身
else:
    raise ValueError(f'unknown objective {predict}')

# ---- 7. 计算 MSE 损失 (仅在 ft 动态通道上) ----
ft_channels = x_start.shape[-1] - f0_channels      # ft 通道数 = C - f0
# 只取最后 ft_channels 个通道计算损失，忽略 f0 (静态形状不需要预测)
terms["mse"] = mean_flat(                          # [B] — 逐样本 MSE
    (target[:, :, -ft_channels:] - model_output[:, :, -ft_channels:]) ** 2
)                                                  # target/output 切片: [B, K, ft]

terms["loss"] = terms["mse"]                       # [B]

return terms
```

训练的结果就是 `DyMeshMMDiT` 学会了生成 `flow` 

> Flow =>也就是 `noise` 到目标 `latent` （运动轨迹）的直线方向。

### EgoTwin

标题：EgoTwin: Dreaming Body and View in First Person

生成第一人称视角的视频

有两个**对齐**的挑战，一个是相机轨迹(决定了相机拍到什么)和人体的头部运动的对齐

二是人体与环境交互的动作和环境变化的对齐（因果交互）

#### 问题定义

输入

- 骨骼序列
- RGB ego View 首帧
- Text Prompt

输出

- 骨骼运动序列 4D  pose sequence
- RGB ego 视频 view sequence

#### Modality Tokenization 不同模态怎么做tokenization

**视频** 使用 3D VAE，使用 $4\times 8\times 8$ 的压缩率

> `3D VAE` 传统的 VAE 处理二维图像，3D考虑了时间维度

**文本** 使用 `T5-XXL` 

##### motion representation 动作表征

> 传统表征：过参数化，记录七个参数
>
> 1. 根部转圈的角速度
> 2. 走位的速度，根部的平面线速度
> 3. 根部的高度(屁股的高度)
> 4. 关节的位置（除了屁股之外，其他关节相对于屁股的位置）
> 5. 局部关节的速度
> 6. 局部关节的旋转
> 7. 脚与地面是否接触

传统表示难以做到与 `ego view` 做对齐，需要 `head-centric` 以头部为中心

> 1. 头部的移动
> 2. 头部的速度
> 3. 头部的旋转角度
> 4. 头部的旋转速度
> 5. 关节的位置  => 关节是以头为参考系的相对表示
> 6. 关节的速度
> 7. 关节的旋转

##### motion tokenization

> causal 1D CNN 处理音频或着视频生成
>
> 普通的 `CNN`  => 模型可以看到 $t-1, t, t+1$ 但是在生成任务中，看不到未来的数据！
>
> 做法：
>
> 左侧填充 $k-1$ 个 0， 每次运算只涉及 $t-k+1, ... , t$ 的数据







我们想要做的是可交互的4D生成。从具体的任务而言，包括应用于具身的任务 一般场景的4D重建 主体的运动4D生成

具身的任务是指对场景，机器人等做4D的表征，让模型了解这一个场景，然后做出决策，例如Dream2Flow、PointWorld

一般场景的4D重建包括 Any4D、D4RT、TrackingWorld。

这两种任务关注于对场景的表征，而主体的运动4D生成更加关注于对于特定主体的表征，往往需要添加骨骼信息，运动序列，而不仅仅是4D点流了，包括 `AnimateAnyMesh` `EgoTwin` `Uni-Inter` `Mo-CapAnything` 
