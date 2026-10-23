# MiniMax-M3.1-Flash：结构与前向张量形状

核验日期：2026-10-23。HF revision：54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e。

证据：固定版本 config、模型卡、本地同版本 safetensors header，以及本地 vLLM 实现（8fae0ff58bae8bde68b52bf3e547432250763e5c）。这是静态结构分析，没有加载全量权重运行模型。以下为不切 TP/EP 的逻辑形状；运行时可融合、分块、量化和分页。

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

这里激活为带 clamp 的 SwiGLU-OAI；配置 alpha=1.702、limit=7。所读 vLLM 配置给 beta 默认值 1.0，其形式为 g*sigmoid(alpha*g)*(u+beta)，g/u 先按实现截断。beta 不是本次 HF config 显式字段，因此这一默认行为来自推理实现。

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