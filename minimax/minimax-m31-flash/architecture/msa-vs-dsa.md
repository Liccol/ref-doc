# MSA 与 DSA：Index Branch 到底选什么、由谁共享

[高清 PNG 4200×3375](msa-vs-dsa.png) · [可编辑 SVG](msa-vs-dsa.svg) · [绘图源码](draw_attention_comparison.py)

![MSA 与 DSA 双分支对照](msa-vs-dsa-preview.png)

两边都按自下向上阅读，并明确保留 **Index Branch** 与 **Main Branch**：输入进入独立索引投影，Index Branch 评分并执行 Top-K；Main Branch 只在选中的历史位置上计算 attention。

| 比较项 | DSA（这里指 DeepSeek-V3.2） | MSA（这里指 MiniMax-M3.1） |
| --- | --- | --- |
| 选择单位 | 单个历史 token | 整个历史 block |
| 索引评分 | 多个索引头点积经 ReLU 后按 query 相关权重聚合 | 每个 GQA 组的 index Q/K 点积，块内取 max |
| 选择结果共享 | 同一 query 的所有 Q heads 使用同一组 token 索引 | 每个 GQA 组独立选块，组内 Q heads 共用 |
| 本模型主分支搭配 | MLA 的 MQA 模式 | GQA：64 个 Q heads / 4 个 KV heads |

图中用 12 个历史位置、每块 2 个 token、每个 Q head 读取 4 个位置做等预算示意。左边各头读相同的离散位置；右边不同 GQA 组读不同的成块位置。图上的数字是位置编号，不是词表 ID，选中集合也不是实测。MSA 的实际配置为 block=128、每组 Top-16 个块并强制包含 local；图中的 2-token 块不能当成模型参数。

“各头共享索引集合”不代表 attention 概率或输出相同；各 Q head 仍有自己的 query。两者都有 Index Branch，因此独立索引器与“先选再算”不是区分二者的充分条件。主要差别在选取粒度及共享范围，不能只凭此图推断速度或精度优劣。

实现阅读入口：ycl `vllm-ascend-private` 的 `vllm_ascend/models/minimax_m3/minimax_m3.py` 中，`MiniMaxM3SparseAttention` 组装模块，`_run_sparse_attention` 执行缓存写入、Indexer 和 SparseImpl 调用；`msa_m3.py` 中分别实现 `AscendMiniMaxM3IndexerImpl` 与 `AscendMiniMaxM3SparseImpl`。具体行号见[前向详解](../model_analyze.md#162-本工作区实现)。

资料来源：[MSA 的架构与 DSA 对比](https://arxiv.org/html/2606.13392v2#S6)、[DeepSeek-V3.2 的 DSA 定义和式 1](https://arxiv.org/html/2512.02556v1#S2.SS1)、[固定 M3.1 配置](sources/config.json)。图为独立结构解释，不复制原图，也不构成运行 trace。

重建：在 `architecture` 目录运行 `python -B draw_attention_comparison.py --png`，需要已安装 Chrome/Edge 和 Pillow。
