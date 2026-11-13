# MiniMax-M3.1-Flash：结构与前向张量形状

核验日期：2026-11-12。本次固定 HF revision：`ce7f4df87d2e05262f5775f373dac4772f134041`。

证据：本次下载的官方固定版本配置、官方 safetensors 元数据抽样，以及当前工作区 vLLM / Ascend 实现。来源、快照与版本见第 16 节。这是静态结构分析，没有加载全量权重运行模型；示例数值用于说明运算，不是该 checkpoint 的真实激活。

配合[现有模型结构图](architecture/README.md)阅读。第 1～7 节是逐层 shape 速查，第 8～10 节解释输入、数值变化和完整算例，第 11～15 节展开缓存、视觉、并行、量化与 DSpark。除特别注明，shape 均为**未切 TP/EP、未量化打包的逻辑形状**。

当前首选图为[单一主线 + 模块展开](architecture/mainline/README.md)：左侧贯穿输入、Tokenizer、Embedding/视觉塔合并、前 3 层、后 57 层及输出，右侧五张小图由对应关键模块引出；以下大分区图作为扩展资料保留。

2026-11-13 图解修订：[整体执行顺序](architecture/detailed/01-overview.png) → [前 3 层 Dense](architecture/detailed/02-dense-layer.png) / [后 57 层 MoE](architecture/detailed/03-moe-layer.png)；[MSA 特征与 FFN 对比](architecture/detailed/05-msa-features-and-ffn.png)单独解释两个不同的“稀疏”。视觉塔先生成同宽视觉向量，再按占位位置与语言向量合成同一序列；进入 MoE 后，两类位置都同时进入 router 和共享分支，不按模态固定分路。

先区分三个对象：**模型参数**是训练得到的权重，推理时通常保持数值和 shape 不变；**激活张量**是输入经过每步运算得到的中间值；**运行时状态**包括 KV cache、位置、块表和路由索引，会随请求推进而更新。下文所谓“轮转”主要指后两者的数据流与状态变化。

## 1. 结论与核心配置

原生图文视频输入、文本输出模型。60 层文本主干全部采用 MSA；前 3 层 Dense FFN，后 57 层 MoE。官方约 428B 总参数、23B 每 token 激活参数。官方未在已读模型卡中给出精确计数口径，不能把本文近似计算当作官方精确统计。

| 项目 | 数值 |
|---|---:|
| hidden_size | 6144 |
| vocab_size | 200064 |
| 文本层数 | 60 |
| Q heads / KV heads / head_dim | 64 / 4 / 128 |
| Q 投影总维度 / K、V 各自总维度 | 8192 / 512 |
| 每 KV 头共享的 Q 头数 | 16 |
| RoPE 维度 / theta | 64 / 10000000 |
| Dense FFN 中间维度 | 12288 |
| 路由专家数 / 每 token top-k | 128 / 4 |
| 共享专家数 / 专家中间维度 | 1 / 3072 |
| MSA 索引 Q 头数 / 索引共享 K 头数 / head_dim | 4 / 1 / 128 |
| 稀疏块大小 / top-k 块 | 128 / 16 |
| 配置最大位置数 | 1048576 |
| Embedding 与 LM Head 共享 | 否 |
| 视觉层数 / 宽度 / 头数 / FFN 宽度 | 32 / 1280 / 16 / 5120 |

注意：hidden_size / num_attention_heads = 96，但该模型显式 head_dim=128，必须使用后者。官方上下文承诺为最多 1M token，不能仅凭 max_position_embeddings 推断服务一定开放 1048576 token。

## 2. 符号与整体流程

B：batch；S：本轮新输入 token 数；P：已缓存长度；T=P+S；N=B*S；M=ceil(T/128)。不等长请求可用 N=sum(S_i)，下列矩形形状只用于解释。

```text
input_ids [B,S]
  -> Embedding [B,S,6144]
  -> 层 0..2：MSA + Dense FFN
  -> 层 3..59：MSA + MoE（4 路由专家 + 1 共享专家）
  -> Final Gemma RMSNorm [B,S,6144]
  -> LM Head [B,S,200064]
  -> 采样 -> 新 token [B,1]
```

单层语义：U=X+MSA(Norm(X))；Y=U+FFN(Norm(U))。所有残差加法两侧形状均为 [B,S,6144]。vLLM 可将 token 轴打包成 [N,6144]，并延后或融合残差加法，不改变这一数学流程。

## 3. Embedding

权重 E=[200064,6144]；input_ids=[B,S]；查表输出 X=[B,S,6144]。这是查表而非构造 one-hot。主干使用 RoPE，不在此处加一个可训练的绝对位置向量表。

## 4. Attention：主分支与 MSA 索引分支

PyTorch 权重矩阵采用 [out_features,in_features]，计算为 X @ W.T。

| 操作 | 权重 | 输出逻辑形状 |
|---|---|---|
| 输入 Norm | [6144] | [B,S,6144] |
| Q projection | [8192,6144] | [B,S,64,128] |
| K projection | [512,6144] | [B,S,4,128] |
| V projection | [512,6144] | [B,S,4,128] |
| Q/K per-head Norm | 各 [128] | 形状不变 |
| Q/K partial RoPE | 每头仅 64 维旋转 | 形状不变 |
| 写入并读取历史 K/V | — | 各 [B,T,4,128] |
| index Q projection | [512,6144] | [B,S,4,128] |
| index K projection | [128,6144] | [B,S,1,128] |
| index Q/K Norm + partial RoPE | 各 Norm [128] | 形状不变 |
| index K cache | — | [B,T,1,128] |
| 按 128-token 块聚合索引分数 | max，带 causal mask | [B,S,4,M] |
| 每个索引头选择块 | top-k=16 | [B,S,4,16] |
| 主稀疏注意力输出 | — | [B,S,64,128] |
| 合并 Q 头 | — | [B,S,8192] |
| O projection | [6144,8192] | [B,S,6144] |

四个索引 Q 头共享一个索引 K 头，但分别产生块选择结果。每个索引头对应一个 KV 组（16 个 Q 头）。sparse_disable_index_value 全为 1，因此不产生额外 index V/O 输出分支。

数学上，索引块分数可以写为：score[b,s,h,m] = max_{j in block(m), j<=query_position} dot(indexQ[b,s,h], indexK[b,j,0])，常数缩放不影响 top-k 排序。先选块，再用主 Q/K/V 对选中块执行 scaled-dot-product attention。主分支缩放为 1/sqrt(128)，softmax 后与 V 相乘。

选中 16 个 128-token 块，对应每组最多 2048 个候选 key 位置；短上下文、causal mask、末块不满会减少有效位置。本地实现将当前局部块强制纳入 top-k 预算（local_block=1，init_block=0）。不能解释成所有头合计只读取相同的 2048 个位置。

逻辑上的选中后分数形状可记为 [B,64,S,Ksel]，Ksel<=2048；实际融合 kernel 不必物化该张量，也不必物化完整 [B,64,S,T]。索引分数同样可以融合计算，不保证实际分配 [B,S,4,T]。

主 Q/K/V 与索引 Q/K 可融合为单次投影：输出 [N,9856]，其中 9856=8192+512+512+512+128。

## 5. Dense FFN：仅第 0、1、2 层

```text
[N,6144]
  -> gate_proj [N,12288] 和 up_proj [N,12288]
  -> SwiGLU-OAI [N,12288]
  -> down_proj [N,6144]
  -> residual add [N,6144]
```

gate/up 权重各 [12288,6144]，down 权重 [6144,12288]。融合 gate_up 输出 [N,24576]，激活后缩回 [N,12288]。

这里激活为带 clamp 的 SwiGLU-OAI；配置 alpha=1.702、limit=7。所读实现给 beta 默认值 1.0：`g'=min(g,7)`，`u'=clamp(u,-7,7)`，输出 `g'*sigmoid(1.702*g')*(u'+1)`。gate 只有上界截断，不能误写成 gate/up 都在 [-7,7] 截断。beta 不是本次 HF config 显式字段，因此这一默认行为来自推理实现。

## 6. MoE：第 3..59 层

```text
X [N,6144]
  ├─ Router [128,6144] -> logits/sigmoid scores [N,128]
  │    -> 带 correction bias 的 top-4 expert ids [N,4]
  │    -> 路由权重 [N,4]
  │    -> dispatch -> expert e 输入 [n_e,6144]
  │         -> w1 gate [n_e,3072]，w3 up [n_e,3072]
  │         -> SwiGLU-OAI [n_e,3072]
  │         -> w2 down [n_e,6144]
  │    -> 按原 token 聚合 [N,6144]
  └─ Shared expert：同样 6144 -> 3072 -> 6144
       -> 与路由结果相加 [N,6144]
       -> residual add [N,6144]
```

专家 e 的逻辑权重：w1/w3=[3072,6144]；w2=[6144,3072]。所有路由专家总 dispatch 数为 sum(n_e)=4N（忽略 kernel padding），并不是每个专家都接收 N 个 token。每 token 经过 4 个路由专家和 1 个共享专家，不是执行全部 128 个路由专家。

在所读实现中，router 为 FP32，sigmoid 路由，correction bias 参与选择；选中权重重新归一化，routed_scaling_factor=2.0。共享专家独立求值后与路由分支相加。

## 7. 输出层与 prefill/decode

最终 Norm：[B,S,6144] -> [B,S,6144]。LM Head 权重 [200064,6144]，与 Embedding 不共享。

完整位置 logits 为 [B,S,200064]；生成服务通常只对选中位置执行 LM Head。普通 prefill 末尾只需 [B,200064]；训练、prompt logprobs、推测解码验证可能需要更多位置。

| 张量 | 无历史 prefill | 普通单 token decode |
|---|---|---|
| ids | [B,S] | [B,1] |
| hidden | [B,S,6144] | [B,1,6144] |
| 本轮 Q | [B,S,64,128] | [B,1,64,128] |
| 本轮 K/V，各自 | [B,S,4,128] | [B,1,4,128] |
| 历史加本轮 K/V，各自 | [B,S,4,128] | [B,P+1,4,128] |
| 本轮 index Q / index K | [B,S,4,128] / [B,S,1,128] | [B,1,4,128] / [B,1,1,128] |
| 下一 token 所用 logits | [B,200064] | [B,200064] |
| 采样结果 | [B,1] | [B,1] |

Prefill 也可能是已有前缀的 chunked prefill，此时 P>0。训练用完整位置 logits 并计算 loss；生成时采样产生的是整数 token ID，不是 6144 维向量。新 ID 在下一轮重新查 Embedding，再经过 60 层；无需把旧 token 的 60 层重新计算一遍。

## 8. 前向调用到底输入什么

以下是概念接口。Transformers 处理器的 batch 维与 vLLM 打包执行接口不同，不能把两套参数原样互传。

| 参数 / 状态 | 逻辑 shape / 类型 | 数值含义和变化 |
| --- | --- | --- |
| `input_ids` | [B,S]，整数；vLLM 为 [N] | 每个值在 [0,200063]；文本、控制符和视觉占位符都在序列中 |
| `inputs_embeds` | [B,S,6144]；vLLM 为 [N,6144]，浮点 | 已查表或已完成视觉替换的向量；使用时可跳过本轮 ID 查表 |
| `attention_mask` | 常见处理器输出 [B,S]，0/1 | 区分 padding 与有效位置；有历史时长度按调用约定覆盖上下文。因果关系还由位置和 attention 元数据约束 |
| `positions` / 文本位置 | 逻辑 [B,S]；vLLM 为 [N]，整数 | 无历史时 0..S-1；已有 P 个 token 时 P..P+S-1；直接用于 RoPE |
| `pixel_values` | [Nv,1176]，浮点 | 处理器已展平的图像 patch；1176=3×2×14×14，不是 [B,3,H,W] |
| `image_grid_thw` | [图像数,3]，整数 | 每行 [gt,gh,gw]，表示时间组数、patch 行数、patch 列数 |
| `pixel_values_videos` / `video_grid_thw` | [Nv_video,1176] / [视频数,3] | 视频经过帧采样、补帧和 patchify 后的输入 |
| `slot_mapping` | [N]，整数 | 本轮 K/V 写到缓存的物理 token 槽；padding 可为无效槽 |
| `block_table` | [B,max_blocks]，整数 | 请求的逻辑历史块到缓存物理块的映射；不是 attention 分数 |
| `seq_lens` / 累计长度 | [B] / 常见 [B+1]，整数 | 每请求总长度、各请求打包 token 的边界 |

Nv 是 merge 前所有视觉 patch token 总数，不能与语言 token 总数 N 混用。不同分支的 grid、槽位和累计长度必须跟对应张量排列一致。vLLM 通常不把 [B,1,S,T] 稠密 mask 传给稀疏 kernel，而是用长度、块表、位置和所选索引表示有效注意力范围。

例如两个请求本轮分别输入 3、2 个 token，打包 `input_ids=[a0,a1,a2,b0,b1]`，N=5；若已有历史长度分别为 10、20，则 `positions=[10,11,12,20,21]`，本轮累计边界可表示为 `[0,3,5]`，总长度为 `[13,22]`。形状 [5,6144] 不意味着 5 个 token 可以互相注意；请求边界仍然隔离。

## 9. Shape 不变时，值究竟如何改变

### 9.1 查表、线性层、reshape 是三种不同的操作

`X[b,s,:]=E[input_ids[b,s],:]` 是取出一行训练好的 embedding。线性层则是 `y_o=sum_i(x_i*W[o,i])+bias_o`，每个输出通道是所有输入通道的加权组合。

例如 Q 的第 h 个头第 d 个通道对应投影的第 `128*h+d` 列：

```text
Q_flat [B,S,8192] -> reshape -> Q [B,S,64,128]
Q[b,s,h,d] = Q_flat[b,s,128*h+d]
```

这一 reshape 不改变数值，只改变索引解释。`transpose/permute` 改变轴顺序；可能只调整 stride，也可能因 contiguous 产生拷贝。矩阵乘法、Norm、RoPE、激活和加法会改变数值。拼接把多个向量排列在一起，不能等同于加法或平均。

### 9.2 Gemma RMSNorm：6144 维或每头 128 维各自归一化

对最后一维长度 D 的向量，所读实现的等价公式为：

```text
r = sqrt(sum(x_i^2)/D + 1e-6)
y_i = (x_i/r) * (1+w_i)
```

Gemma 的可训练参数是 w，实际增益为 `1+w`。它不减均值。层输入 Norm 的 D=6144；Q/K per-head Norm 的 D=128，每个 token、每个 head 独立计算 RMS，且同类 head 共享长度为 128 的 Norm 权重。本层四套 Q、K、index Q、index K Norm 各自有独立权重。

教学例：x=[3,4]、w=[0,0]，忽略 epsilon，RMS=sqrt(12.5)，得到约 [0.848528,1.131371]。shape 仍是 [2]，数值已改变；真实模型不会因此变成二维。

### 9.3 文本 partial RoPE：位置在数值中旋转，轴长度不增加

文本主 Q/K 与 index Q/K 都是 head_dim=128，只对前 64 维旋转；后 64 维保持原值。按本地 NeoX 风格，在前 64 维中将第 i 维与第 i+32 维配对，i=0..31：

```text
theta_i(p) = p * 10000000^(-2*i/64)
a' = a*cos(theta_i) - b*sin(theta_i)
b' = b*cos(theta_i) + a*sin(theta_i)
```

Q 与 K 使用各自 token 的位置 p；V 不做这一步旋转。p=0 时 cos=1、sin=0，这一步不改变值；不同位置产生不同旋转，使 Q/K 点积带有相对位置信息。推理位置推进的是 p，模型的 rope_theta、rotary_dim 和投影权重不会随 token 改变。

### 9.4 MSA 的两次“选取”与一次 softmax

Indexer 使用 **index Q/K** 对历史块评分并选块；主 attention 使用独立的 **主 Q/K/V** 计算最终输出。Indexer 分数不直接变成主 attention 的概率。

```text
I[b,s,h,m] = max_{j 属于块 m 且 j<=p_s} <indexQ[b,s,h], indexK[b,j,0]>
J[b,s,h,:] = TopK(加入 local/init 优先级并屏蔽无效块后的 I)
g(q_head) = floor(q_head/16)
L[b,q_head,s,j] = <Q[b,s,q_head],K[b,j,g(q_head)]>/sqrt(128)
A = softmax(L，在该 head 选中块内的有效 key 位置上归一化)
O[b,s,q_head,:] = sum_j A[b,q_head,s,j] * V[b,j,g(q_head),:]
```

上式省略 index 排序所用常数缩放。对当前查询，未来位置即使位于已选块中，也必须屏蔽。索引头 h=0..3 分别服务 KV 组 0..3，主 Q 头 0..15、16..31、32..47、48..63 分别使用对应组选择的块。同组不同 Q 头共享块集合和 K/V，但 Q 不同，softmax 权重及输出也不同。

数值教学例：把块大小临时缩到 2、Top-K 缩到 2，仅用于说明。三个块内 index 点积分数为 `[1,4]`、`[3,2]`、`[0,1]`，max 得 `[4,3,1]`。若块 2 是必须保留的 local 块，最终选择块 0、2，块 1 被排除。实际模型使用 block_size=128、Top-K=16。

再假设某主 Q 头在所选有效位置上的缩放后 logits 为 `[0,ln(3)]`，softmax 得 `[0.25,0.75]`；若对应 V 的某两通道为 `[2,0]`、`[0,4]`，该头输出这两通道为 `[0.5,3]`。它是 V 的加权和；O projection 随后再把所有 64 个头混合为 6144 维。

### 9.5 MoE：选择分数与输出权重分开

按本地 bias-corrected sigmoid router 的语义，对 token n：

```text
z[n,e] = <X[n,:], W_router[e,:]>                  # [128]
s[n,e] = sigmoid(z[n,e])                         # [128]
ids[n,:] = Top4_indices(s[n,:] + correction_bias) # [4]
r[n,k] = s[n,ids[n,k]] / sum_j s[n,ids[n,j]]      # [4]
R[n,:] = 2.0 * sum_k r[n,k] * Expert_ids[n,k](X[n,:])
F[n,:] = R[n,:] + SharedExpert(X[n,:])            # [6144]
```

correction bias 参与“选谁”，输出权重用被选专家的**原始 sigmoid 分数**归一化，不能把加 bias 后的数值直接拿来加权。r 的和为 1；等价地把 2.0 乘到权重后，其和为 2。共享专家输出不再乘这个 routed scaling factor。

示例：某 token 被选中的 expert IDs 为 `[7,12,31,80]`，原始 sigmoid 分数为 `[0.8,0.6,0.4,0.2]`，归一化得到 `[0.4,0.3,0.2,0.1]`。若四专家在某个输出通道的值分别为 `[10,20,-10,0]`，路由分支该通道为 `2*(4+6-2+0)=16`；共享专家该通道若为 3，则 FFN 输出为 19。若 attention 后残差该通道为 5，则该层完整输出为 24。

Dispatch 并不修改原 token 的含义：为每个 token 建立 4 条带 expert ID 和权重的任务，按 expert 分组后计算，再按原 token ID scatter/gather 回来。专家内部的 padding、排序和 EP 通信可以改变物理行次序，但聚合后必须恢复原始 token 顺序。

### 9.6 残差在实现中的“延后相加”

逻辑上始终是 `U=X+Attention(Norm(X))`，`Y=U+FFN(Norm(U))`。本地 decoder 为融合 Norm，可能返回 `(hidden_states=FFN_output, residual=U)` 两个 [N,6144] 张量；下一层 input norm 才执行二者相加并归一化。最终 Norm 也处理这对状态。不能把中间的 `hidden_states` 单独当成已完成残差加法的 Y，也不能额外再加一次 residual。

## 10. 一次完整 prefill，再推进两轮 decode

### 10.1 文本例：B=2，每请求 S=256，无历史

此例无 padding，N=512；维度是逻辑值，不要求实际一次分配所有中间张量。

| 步骤 | 张量 shape | 说明 |
| --- | --- | --- |
| Token IDs / positions | [2,256] / [2,256] | 两请求各自位置为 0..255 |
| Embedding / 层输入 Norm | [2,256,6144] | 打包为 [512,6144] |
| 主 QKV 融合投影 | [512,9216] | split 为 8192、512、512 |
| index QK 融合投影 | [512,640] | split 为 512、128；与主投影再融合时总宽 9856 |
| Q；K；V | [2,256,64,128]；K/V 各 [2,256,4,128] | Q/K Norm + RoPE 后 shape 不变 |
| index Q；index K | [2,256,4,128]；[2,256,1,128] | 写 index K cache |
| 主/索引缓存的有效长度 | 各请求 256 | 每层独立维护自己的缓存 |
| index 块分数 | 概念 [2,256,4,2] | 每请求历史总共 2 块；早期 query 看不到未来块 |
| Top-K 结果 | 概念容量 [2,256,4,16] | 只有 1～2 个有效块，其他槽屏蔽；实际输出轴顺序按 kernel |
| 主 attention 输出 / O 投影 | [512,8192] / [512,6144] | 同宽残差相加 |
| 第 0～2 层 Dense gate/up | 各 [512,12288] | 融合输出 [512,24576]；激活后 [512,12288] |
| 第 3～59 层 router | [512,128] | expert IDs 和权重各 [512,4] |
| 一层 MoE 总 dispatch | 2048 条路由任务 | 每个专家输入 [n_e,6144]，sum(n_e)=2048 |
| 共享专家 gate/up | 各 [512,3072] | 每 token 都计算共享分支 |
| 完成 60 层 / Final Norm | [512,6144] | 每层宽度相同，数值和缓存内容不同 |
| 选取两个请求的最后位置 | [2,6144] | 只为下一 token 计算 logits |
| LM Head / 采样 | [2,200064] / [2,1] | 得到每请求第 257 个 token 的 ID |

### 10.2 Decode 第 1、2 轮：token 和缓存的时序

| 状态 | prefill 完成 | decode 第 1 轮 | decode 第 2 轮 |
| --- | --- | --- | --- |
| 送进模型的 token | 原 prompt 256 个 | 上轮采样的第 257 个 | 上轮采样的第 258 个 |
| 每请求 P / S | 0 / 256 | 256 / 1 | 257 / 1 |
| 本轮 positions | 0..255 | [256] | [257] |
| 本轮 hidden（打包） | [512,6144] | [2,6144] | [2,6144] |
| 每层 Q（打包） | [512,64,128] | [2,64,128] | [2,64,128] |
| 每请求缓存有效长度 | 256 | 257 | 258 |
| 每请求逻辑稀疏块数 | 2 | 3 | 3 |
| 当前 local 块号（从 0 起） | 对末位置为 1 | 2 | 2 |
| 本轮采样产物 | 第 257 个 ID | 第 258 个 ID | 第 259 个 ID |

**刚采样出的 token 尚未进入 KV cache**；要等下一次作为 input 经过该层投影后才写入该层 cache。decode 不把缓存切成只留当前 Top-16 块；本轮未选中的旧块通常仍保留，后续查询可以再次选中。

## 11. KV cache、块索引和显存维度

### 11.1 逻辑 cache 容量与当前计算量

不分片时，每层、每历史 token 存储：K=4×128、V=4×128、index K=1×128，共 **1152 个标量**；不缓存所有历史 Q。若三类都用 BF16，纯数据为 2304 bytes/token/layer；60 层为 138240 bytes/token，即 **135 KiB/token**。

单请求 T=131072 时，纯 BF16 三类 cache 合计 16.875 GiB；T=1048576 时为 135 GiB。这不含页尾浪费、对齐、元数据、量化 scale、权重、工作区、视觉中间值和其他请求，也不代表特定部署的实测显存。不同 KV dtype 和分片方式会改变容量。

Top-K 稀疏主要减少主 attention 读算的位置数，不会仅凭 K=16 就把历史存储降为常数；indexer 还需要访问历史 index K 来决定选块。

### 11.2 分页布局和地址轮转

数学上 K/V 各 [B,T,4,128]，index K 为 [B,T,1,128]。实际 cache 可按 `[num_physical_blocks,cache_block_size,heads,head_dim]` 或带 K/V 轴的等价布局存储，也可能转置、padding 或按后端打包。**MSA 的 128-token 选择块与 cache allocator 的物理页大小属于不同概念**；在某些配置中数值相同，也不能在所有后端中直接等同。

若教学例中物理页大小也取 128，请求逻辑块表为 `[9,2,15]`，那么绝对位置 256 对应逻辑块 2、块内 offset 0；物理页为 15，扁平槽号为 `15*128+0=1920`。下一 token 位置 257 写槽 1921。块表 `[9,2,15]` 的值是地址编号，与内容相似度或时间先后大小无关。请求结束后页可回收，被别的请求复用。

模型每一层的 K/V 值都不同。共享分页分配方案不等于不同层共享同一份 K/V 数据。TP 复制 KV heads 的部署还会产生物理重复，不能把单 rank 元素数简单乘 TP 后当作唯一逻辑 cache 容量。

## 12. 图像、视频：从像素到 [N,6144]

### 12.1 像素预处理和 patch 排列

以**预处理后尺寸确为 672×672 的一张静态图**为例；dynamic resolution 会根据输入调整尺寸，因此这不是所有图片的固定形状。静态图复制/补齐到 temporal_patch_size=2 帧，同一时间组内形成 2×14×14 的 RGB patch：

```text
RGB 图像 [672,672,3]
 -> resize / rescale / normalize / temporal padding
 -> 按 2 帧、14×14 空间 patch 分组
 -> grid_thw = [[1,48,48]]
 -> Nv=1*48*48=2304
 -> pixel_values [2304,3*2*14*14] = [2304,1176]
```

官方处理器均值为 `[0.48145466,0.4578275,0.40821073]`，标准差为 `[0.26862954,0.26130258,0.27577711]`。对通常 [0,255] 像素按处理器默认 rescale 后，各通道做 `(pixel/255-mean)/std`。patchify 保留这些像素的数值但重排位置；归一化改变数值。处理器会把邻接 2×2 patch 排在可连续 merge 的顺序，不能对任意 raster-order 向量直接 reshape 并假设邻接关系正确。

### 12.2 Patch Embedding 和 ViT ×32

| 操作 | 权重（逻辑） | 上述图像的输出 |
| --- | --- | --- |
| patch 输入 reshape | — | [2304,3,2,14,14] |
| Conv3d，kernel=stride=(2,14,14)，无 bias | [1280,3,2,14,14] | [2304,1280,1,1,1] -> [2304,1280] |
| Pre LayerNorm | gain/bias 各 [1280] | [2304,1280] |
| 进入 encoder 的存储布局 | — | [2304,1,1280]，中间轴为打包 batch=1 |
| 单层 LN1 + QKV linear | 融合 [3840,1280]，bias [3840] | [2304,1,3840] |
| 拆 Q/K/V | 各逻辑 [1280,1280] | 各 [1,2304,16,80] |
| Q/K 3D partial RoPE | 无可训练位置表 | shape 不变；V 不旋转 |
| 视觉 attention | 每头缩放 1/sqrt(80) | [1,2304,16,80] |
| 合并 heads + out projection + residual | [1280,1280]，bias [1280] | [2304,1,1280] |
| LN2 + fc1 + GELU | [5120,1280]，bias [5120] | [2304,1,5120] |
| fc2 + residual | [1280,5120]，bias [1280] | [2304,1,1280] |
| 重复 32 层并移除单例轴 | — | [2304,1280] |

视觉 LayerNorm 与文本 GemmaRMSNorm 不同：它减均值，除标准差，epsilon=1e-5，带 gain 和 bias。视觉 MLP 使用 GELU，不使用文本的 SwiGLU-OAI。

按当前 `vision_tower.py` 实现，head_dim=80，3D RoPE 给时间、高度、宽度**各 26 维**，总共旋转 78 维，剩余 2 维不变；cos/sin 各 [Nv,39]。26 来自 `2*((80//3)//2)`。这属于实现细节，不能仅由 config 的 `rope_mode=3d` 推断为每轴平均分到完整 80 维。

视觉 attention 在同一图片/视频片段内为双向 attention，使用分段累计长度隔离不同项目，不使用语言的自回归 causal mask，也不使用主干的 Top-16 MSA。上述单图的 attention score 在数学上为 [1,16,2304,2304]，Flash Attention 可不物化完整矩阵。最后使用 raw last encoder hidden state；当前封装明确关闭没有 checkpoint 权重的 post LayerNorm。

### 12.3 Projector 在前，merger 在后

```text
ViT 输出                        [2304,1280]
projector linear_1 + bias       [2304,6144]   W=[6144,1280]
GELU + linear_2 + bias          [2304,6144]   W=[6144,6144]
相邻 2×2 token 拼接             [576,24576]   24576=4*6144
merge linear_1 + bias + GELU    [576,6144]    W=[6144,24576]
merge linear_2 + bias          [576,6144]    W=[6144,6144]
```

所有四个 bias 都是 [6144]。拼接不丢元素：2304×6144=576×24576；后续可训练线性映射才完成空间压缩，不是对四 patch 求均值。最终每图的视觉 token 数为 `gt*gh*gw/4`。配置中的 `image_seq_length=576` 恰好对应此例；动态分辨率下应以 grid 计算实际数量。

### 12.4 合入文本与视频示例

处理器在语言输入中展开相应数量的占位符；image token ID 为 200025，video token ID 为 200026。先得到输入 embedding，再将对应占位位置替换成 [Nv/4,6144] 视觉向量，保持语言序列 shape [B,S_total,6144]。需保证占位位置数与视觉向量数一致；边界符、时间戳等控制 token 另计。

例如 100 个非视觉 token（已经包含所需控制符）加本例 576 个图像位置，则 S_total=676，输入主干为 [1,676,6144]。视觉信息随后参与同一条语言自注意力序列，没有额外 cross-attention。生成文本时通常不每轮重跑相同图像的 ViT，视觉前缀的影响保留在各层 KV cache 中。

视频例：假设采样/补齐后 8 帧、每帧 672×672，则 gt=8/2=4、Nv=4×48×48=9216，`pixel_values_videos=[9216,1176]`，视觉压缩输出 [2304,6144]。视频帧数从 8 变成 4 个时间组是 temporal patch embedding 的结果；后续 2×2 merger 只压缩空间。当前 ViT 实现按 `vision_segment_max_frames=4` 对时间 grid 的组数限制片段长度；此例 gt=4 为一个片段。更长视频会按实现拆片段并用对应的 cu_seqlens 隔离，不能把任意长视频看成一个无限长的视觉 attention 序列。

## 13. TP / EP 后为什么看到的 shape 不一样

TP 分割矩阵通道，EP 分割专家集合，SP 分割 token 行；三者作用的轴不同。下面以 TP=4、无 SP、KV heads 与 index Q heads 正好可均分为例，N 仍为本轮所有有效 token 数。

| 模块 | 全局逻辑 | 单 TP rank 示例 |
| --- | --- | --- |
| Q 输出 / 权重 | [N,8192] / [8192,6144] | [N,2048] / [2048,6144]，16 Q heads |
| K 或 V 输出 / 权重 | [N,512] / [512,6144] | [N,128] / [128,6144]，1 KV head |
| index Q 输出 / 权重 | [N,512] / [512,6144] | [N,128] / [128,6144]，1 index Q head |
| index K 输出 / 权重 | [N,128] / [128,6144] | [N,128] / [128,6144]，共享 index K head 通常复制 |
| 五路融合投影宽度 | 9856 | 2048+128+128+128+128=2560，**不是 9856/4** |
| O projection 权重 | [6144,8192] | [6144,2048]，各 rank 得到 [N,6144] 部分和，再归约 |
| Dense gate/up 输出 | 各 [N,12288] | 各 [N,3072] |
| Dense down 权重 | [6144,12288] | [6144,3072]，输出归约为 [N,6144] |

TP>4 时，KV 头数已不足以每 rank 分到独立整数头，具体实现需要复制 KV heads；index heads 也可能复制或在评分阶段通信。不能照搬“所有维度除 TP”。某些优化还沿历史块维分割 index cache；上表是基础投影分片示例，不描述所有后端的 cache 分布。

仅 EP=8、无冗余专家时，每 rank 持有 128/8=16 个路由专家；每个专家仍有完整 6144→3072→6144 的逻辑矩阵。All-to-all 按 expert 所在 rank 发送输入并回收输出；收发 token 数由路由决定，不保证均为 4N/8。TP 与 EP 组合、共享专家部署、EPLB 复制和 SP padding 都需以实际配置计算。

Embedding / LM Head 还可沿词表轴分片，并对词表做实现要求的 padding；看到本地词表行数时，应区分逻辑词表 200064 与 padding 后大小。

## 14. 公开量化权重的存储 shape 与 dtype

本次固定版本的 config 使用 `quant_method=mxfp8`，但路由专家另指定 `moe_quant_algo=NVFP4`、`group_size=16`。因此不能把整个仓库简单称为“所有权重都是 FP8”。以下来自**本次实际读取的 safetensors header**，不是仅从文件名推断；只读取元数据，没有下载权重数据区。

| 参数样本 | 数学 shape | 文件中 dtype / shape |
| --- | --- | --- |
| Embedding、LM Head | [200064,6144] | BF16，[200064,6144]；为两个独立参数 |
| 层 0 Q weight | [8192,6144] | F8_E4M3，[8192,6144] |
| 层 0 Q `weight_scale_inv` | 每 32 个输入通道一组 | U8，[8192,192]，192=6144/32 |
| 层 0 O weight / scale | [6144,8192] | F8_E4M3 [6144,8192] / U8 [6144,256] |
| 层 0 Dense gate weight / scale | [12288,6144] | F8_E4M3 [12288,6144] / U8 [12288,192] |
| 层 3 router weight | [128,6144] | F32，[128,6144] |
| 层 3 expert 0 w1/w3 `weight_packed` | 每个 [3072,6144] | U8，[3072,3072]，一个 byte 打包两个 FP4 权重 |
| 层 3 expert 0 w1/w3 `weight_scale` | 每 16 个输入通道一组 | F8_E4M3，[3072,384] |
| 层 3 expert 0 w2 `weight_packed` | [6144,3072] | U8，[6144,1536] |
| 层 3 expert 0 w2 `weight_scale` | 每 16 个输入通道一组 | F8_E4M3，[6144,192] |
| expert input/weight global scale | 标量 | F32，[] |
| 视觉 Conv3d weight | [1280,3,2,14,14] | F32，[1280,3,2,14,14] |
| projector / merge 首层 weight | [6144,1280] / [6144,24576] | BF16，同逻辑 shape |

这些条目的完整名称和所属 shard 见 [header 抽样快照](architecture/sources/weight-header-samples.json)。MXFP8 的 U8 scale 是编码后的缩放信息，不能把原始 byte 直接当普通实数去乘；NVFP4 还组合 block scale 与 global scale，具体数值恢复须遵守对应量化格式。

概念过程是“激活动态量化 / 读取权重编码与 scale → 量化 GEMM 与缩放 → 后续 Norm、激活或累加”。逻辑上输出仍是 [N,out_features]；量化主要改变表示方式、精度和物理存储，加载器还可进一步转置/重排。文件 header dtype、设备参数 dtype、算子输入/累加/输出 dtype 可能不同，不能只看根配置 `torch_dtype=bfloat16` 就断言每一步都是 BF16。

历史 A5 BF16 验收是另一份本地资产。本次公开量化权重的 header 核验不构成重新运行公开量化模型或验证其精度。

## 15. DSpark：公开附带的草稿分支

已有主结构图以关闭投机解码的 60 层 target 为范围。公开仓库另有 `dspark/`，需要单独区分：它不是第 61～65 层的普通串行延长，也不将主模型层数改为 65。

固定版本 [DSpark config](architecture/sources/dspark-config.json) 给出：5 层 draft、hidden=6144、64/4 heads、head_dim=128、Dense FFN intermediate=12288、sliding_window=4096、block_size=7、Markov rank=256、noise token ID=200058；读取 target 层号 `[3,17,31,45,59]`。

以下辅助权重 shape 由该配置及本地适配器的 `expected_shapes` / BF16 header fixture 交叉解释，**没有对公开 DSpark 数据文件做本次 header 核验**：

| 数据 / 参数 | 逻辑 shape | 含义 |
| --- | --- | --- |
| 对齐位置的 5 层 target hidden | 各 [N,6144] | 捕获的是选定层的上下文特征，精确时点按推理实现 |
| 拼接与融合 fc | [N,30720] -> [N,6144]；W=[6144,30720] | 30720=5×6144，压回 draft hidden width |
| 每层 draft Q/K/V | Q [N,64,128]，K/V 各 [N,4,128] | 5 层 draft 采用滑窗 attention；不是 target 的 MSA 配置 |
| 每层 draft FFN | [N,6144] -> [N,12288] -> [N,6144] | gate/up 融合宽度 24576 |
| Markov 辅助权重 | 两个 [200064,256] | 词表与 rank=256 的辅助头参数；不能当成主干 hidden size |
| confidence head | W=[1,6400]，bias=[1] | 6400=6144+256，输出每位置 1 个 confidence 值 |

block_size=7 是草稿生成/验证块的配置，不等于 7 个草稿层，也不保证每步接收 7 个 token。启用后由 draft 提议候选，target 按协议验证、接受部分候选并推进状态；未接受部分需要相应回退。具体候选布局、block 内是否含 anchor 和 cache 提交时序取决于 DSpark 实现，不能仅由这份 config 还原为固定的 [B,7] 调用流程。第 10 节的普通单 token decode 假设关闭这一路径。

## 16. 来源、可复核范围与版本

### 16.1 官方固定版本

本次在线核验统一固定 revision `ce7f4df87d2e05262f5775f373dac4772f134041`，避免 main 后续变化影响本文。

| 来源 | 用途 | 本地副本 |
| --- | --- | --- |
| [官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/README.md) | 约 428B/23B、输入输出、MSA 与 DSpark 范围 | 通过官方链接查阅 |
| [config.json](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/config.json) | 结构、路由、视觉、量化配置 | [快照](architecture/sources/config.json) |
| [preprocessor_config.json](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/preprocessor_config.json) | 像素均值、标准差、patch 和 merge 配置 | [快照](architecture/sources/preprocessor_config.json) |
| [image_processor.py](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/image_processor.py) | 1/255 rescale、单图补帧、相邻 patch 重排与 flatten | 只读取源码核对，不执行远程代码 |
| [dspark/config.json](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/dspark/config.json) | 草稿分支配置 | [快照](architecture/sources/dspark-config.json) |
| [权重索引](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/model.safetensors.index.json)及 shard header | 通过 HTTP Range 只读 safetensors 长度和 JSON header；核验典型 dtype / 存储 shape | [73 项抽样元数据](architecture/sources/weight-header-samples.json) |

原架构说明固定于 `54e4e69...`；本次在当前 revision 重新检查结构关键字段，本文使用当前快照。已有图仍保留原版本的绘图来源，不将其历史 NPU 验证记录改写为本次运行结果。

### 16.2 本工作区实现

vLLM HEAD=`ced6857afa0ea7b2e3f0846a62e1394e90f15607`；vLLM Ascend HEAD=`5666e6616ecc0155fb9ae79d861eb9f9781e6f31`。以下相对链接从本文位置指向同一工作区的仓库；它们不是公开 HF 自带的模型实现。

仓库来源：语言主干和 Ascend MSA 为 [ycl-ai-infra/vllm-ascend-private](https://github.com/ycl-ai-infra/vllm-ascend-private/tree/5666e6616ecc0155fb9ae79d861eb9f9781e6f31)；视觉塔等共用组件为 [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/ced6857afa0ea7b2e3f0846a62e1394e90f15607)。M3.1 使用 M3 系列共用目录 `minimax_m3`：官方 `architectures=MiniMaxM3SparseForConditionalGeneration` 经 [Ascend 模型注册](../../../vllm-ascend-private/vllm_ascend/models/__init__.py)映射到该目录；Decoder 读取 M3.1 配置中全为 1 的 `sparse_attention_freq` 和前三项为 0 的 `moe_layer_freq`，选择 60 层 MSA、前 3 Dense / 后 57 MoE。此项为静态注册与配置分支核验，不是本次实际运行的 dispatch trace。

- [Ascend 语言模型](../../../vllm-ascend-private/vllm_ascend/models/minimax_m3/minimax_m3.py)：`MiniMaxM3SparseAttention`、`MiniMaxM3MLP`、`MiniMaxM3MoE`、`MiniMaxM3DecoderLayer`；投影、激活、路由参数和延后残差。
- [Ascend MSA 实现](../../../vllm-ascend-private/vllm_ascend/models/minimax_m3/msa_m3.py)与 [MSA Triton kernels](../../../vllm-ascend-private/vllm_ascend/models/minimax_m3/ops/msa_m3_triton.py)：cache、index score、local/init 优先级与 Top-K。
- [视觉编码和融合](../../../vllm/vllm/models/minimax_m3/common/vision_tower.py)：`MiniMaxVLPatchEmbed`、`MiniMaxVLAttention`、`MiniMaxVLVisionModel`；3D RoPE 的 26/26/26、projector 在 merger 前及 post norm 行为。
- [多模态预处理接入](../../../vllm/vllm/models/minimax_m3/common/mm_preprocess.py)与 [Ascend 多模态封装](../../../vllm-ascend-private/vllm_ascend/models/minimax_m3/minimax_m3_vl.py)：grid、占位符展开、视觉输入解析。
- [GemmaRMSNorm](../../../vllm/vllm/model_executor/layers/layernorm.py)与 [bias-corrected 路由](../../../vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py)：`1+w` 归一化、原始 sigmoid 分数的重归一化。
- [DSpark 适配](../../../vllm-ascend-private/vllm_ascend/patch/minimax_m3_dspark.py)与 [BF16 DSpark header fixture](../../../vllm-ascend-private/tests/ut/models/minimax_m3/fixtures/minimax_m31_bf16_dspark_header.json)：草稿参数 shape 的补充依据。

### 16.3 结论边界

结构维度与采样 header 经过静态核对；公式是实现的逻辑解释，不是浮点逐位等价声明。文中 scalar 示例、672×672 图像和 TP=4 算例是明确给定条件下的推导；没有把它们当作模型实测激活、实际图像处理日志或硬件 profiling。真实激活值需要指定 prompt/图像、权重、精度、并行和后端后抓取，单凭 config 无法给出。
