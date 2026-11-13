# MiniMax-M3.1-Flash：一条主线与关键模块展开

[完整高清 PNG 5100×4890](../minimax-m31-detailed-architecture.png) · [可编辑 SVG](../minimax-m31-detailed-architecture.svg) · [完整 shape / 数值流转详解](../../model_analyze.md)

![主线及模块放大图](preview.png)

左侧实线是端到端计算：输入 → 文本 Tokenizer / 视觉预处理 → Embedding / 视觉塔 → 按占位位置合并 → 前三层 → 后 57 层 → Final GemmaRMSNorm → LM Head → 采样 → 文本输出。输入阶段两路在进入语言主干前合并，后续只有一条主线。

前 3 层和后 57 层各用一个代表层框，并标记 ×3 / ×57；框内只列 GemmaRMSNorm、Attention、Dense FFN / MoE 及两次残差。结构相同不表示各层共享权重。

虚线与编号连接对应的小图，表示模块放大说明，不是额外执行分支：

| 编号 | 模块小图 | 展开内容 |
| --- | --- | --- |
| 主线 | [PNG](main-path.png) / [SVG](main-path.svg) | 输入到输出的完整路径 |
| ① | [视觉塔](01-vision.png) / [SVG](01-vision.svg) | Patch Embed、ViT ×32、projector、merger、返回视觉向量 |
| ② | [GemmaRMSNorm](02-gemma-rmsnorm.png) / [SVG](02-gemma-rmsnorm.svg) | RMS 归一化、1+w 增益、归一化轴和 shape |
| ③ | [Attention / MSA](03-attention-msa.png) / [SVG](03-attention-msa.svg) | 主 QKV、index QK、Norm/RoPE、缓存、Top-K 选块、主 attention 和输出投影 |
| ④ | [Dense FFN](04-dense-ffn.png) / [SVG](04-dense-ffn.svg) | gate/up、SwiGLU-OAI、down |
| ⑤ | [MoE](05-moe.png) / [SVG](05-moe.svg) | router、dispatch、4 个路由专家、共享专家、combine 与结果相加 |

图上 shape 为未分片的逻辑维度。N 是合并后的有效语言序列长度；Nv 是 merger 前视觉 patch 数。GemmaRMSNorm 的 w 为增益参数；Dense / Expert 的 SwiGLU-OAI 使用上界截断 g′=min(g,7)、u′=clamp(u,−7,7)。完整参数 shape 与实现依据放在文字详解中。

布局参考 [CalvinXKY/InfraTech MiniMax-M2.5 架构图](https://github.com/CalvinXKY/InfraTech/blob/main/models/minimax_m_2_5/minimax_m_2_5_architecture.jpg)，只参考主线与模块放大的组织方式。M3.1 的维度和模块来自[固定版本配置与 header](../sources/README.md)、ycl Ascend 适配及上游 vLLM 共用视觉实现；不是对 M2.5 参数进行替换后推测，也不是本次实际运行 trace。

重建：在 `architecture` 目录运行 `python -B draw_mainline.py --png`。SVG 保留文字和路径，可编辑；PNG 依赖已安装 Chrome/Edge 与 Pillow。绘图原语由 `diagram_primitives.py` 提供。
