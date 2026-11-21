# 标准低秩与跨 TP 继续定位

**状态：原 token 硬指标仍未通过，本轮没有可合入的生产修复。** 按要求采用标准低秩，未使用 merged 目标层缓存。新做了两种精度改动的 NPU 因果对照、TP1/TP2 标准低秩对照，以及真实输入的单层和 HCCL 归约实验。失败的改动只保留在测试 worker extension 中。

[原始请求、参考输出、批次映射及诊断源码](standard-low-rank-evidence.tar.gz)，[SHA256](SHA256SUMS)。45个文件，无模型 tensor、内网地址或凭据。`summary.json` 的状态为 `not_accepted`，分别记录原生和 BI 实验；源码哈希及测试扩展函数 AST 哈希用于核对复现版本。

诊断工具和报告已推送个人分支的 [提交 `a2f6c9208`](https://github.com/Liccol/vllm-ascend/commit/a2f6c92082d951f10d3b94eb253e4437d4d9dbeb)。该提交只增加测试与文档，没有修改生产执行路径。

## 1. 条件与未改变的内容

vLLM 固定 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，Ascend 在既有 `b9db9e22e` 上运行默认 `low_rank`。本轮未改 vLLM 或生产 LoRA 源码，未升级依赖。

仍是原来的四层真实 BF16、Q/K/V/O、rank8/alpha16、独立 A/B、标准差0.1、四条固定 prompt、32个生成 token。参考与服务开启 BI、eager，关闭 prefix cache、chunked prefill、SP/FlashComm，block size128。没有改变权重、输入、生成长度或 token/finish_reason 判定。

## 2. 标准低秩的两种精度改动

`standard_controls.py` 通过 worker 生命周期注入测试控制；所有模式都按 `X @ A.T @ B.T` 计算，不构建合并目标权重。

- `original`：原始标准低秩，作为本轮复现基准。
- `post_reduce`：O 投影先归约基座输出和 FP32 shrink，再 expand 并相加；QKV 保持原路径。
- `fp32_local`：QKV/O 的基座和低秩增量在 FP32 中相加；O 的局部输出用 FP32 归约，到末尾才转 BF16。它会增加临时缓冲和计算成本，属于精度控制，不是性能优化。

| 路径 | TP2/DP2/EP 对同拓扑 merged | TP2 混合服务 | TP1/DP1 对同拓扑 merged | TP1 混合服务 | 标准低秩 TP1 vs TP2 |
| --- | --- | --- | --- | --- | --- |
| 原路径 | 3/8 | 36/36 | 3/8 | 36/36 | 3/8 |
| O 归约后 expand | 4/8 | 32/36 | 4/8 | 36/36 | 3/8 |
| FP32 局部累加及归约 | 3/8 | 33/36 | 5/8 | 36/36 | 3/8 |

所有模型比较均为完整32-token序列。最后一列比较相同模式的标准低秩服务，不是与旧 merged 参考混比；整模型拓扑也改变了 DP 和 EP world size，不能把全部差异都归因于纯 TP。两个控制均未达到原要求，并在 TP2 出现服务回归，因此没有合入生产逻辑。

原路径的真实混批证据：DP0 两个 TP worker 各233条 mapping 更新、6次同 batch 含 base/A/B；DP1 各232条更新、6次同 batch 含 base/A/B。两个 TP rank 不能相加计作独立 batch。

## 3. 为什么单纯提高低秩精度不够

第一层真实 Q/O 权重及同一份真实输入的独立对照，先在 CPU FP64 中计算基座与低秩增量，最后只舍入一次。以 adapter-B 为例：

| 目标层 | 与 FP64 merged GEMM 最后转 BF16 的相对 L2 差异 | 不同元素 |
| --- | --- | --- |
| Q | 0.00169022 | 8,719/32,768 |
| O | 0.00199395 | 8,731/24,576 |

这些值不依赖生产 Punica kernel 的计算误差。两个 adapter 的 Q/O 都复现差异；这说明去掉基座输出的中间舍入仍不能消除 BF16 合并权重舍入造成的差异。

```text
标准低秩理想值： X W + (X A.T) B.T
BF16 merged：   X BF16(W + B A).T
```

两套 BF16 权重逐元素检查均符合保存时的合并公式。实际 `BF16(W + BA) - W` 的64×64子矩阵秩为64，而原更新秩为8。合并舍入后的有效更新不是同一个 rank8 矩阵；修正 alpha 或重新切片不能令它们对任意输入完全等价。

这不构成“任何实现都不可能碰巧通过这8条 prompt”的证明，也不能凭矩阵不等就替代 token 测试；完整请求的失败已经在上表单独保留。结论是不能把逐 token 相同作为两种 BF16 求值方式对任意输入的通用保证。

PyTorch 2.10 的 [数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) 也说明，浮点运算次序、批处理和切片方式变化可能改变数值结果；其说明本身不替代本次 NPU 实测。

## 4. 跨 TP 的独立证据

`tp_precision.py` 使用两张真实 NPU、HCCL all-reduce 和共同的第一层 O 输入，只比较投影，不执行 attention 或 MoE。它将纯 TP 局部计算/归约与整模型 DP、EP 差异分开。

按服务使用的 AscendC BI 算子和 deterministic level3 复核：

| HCCL 两卡路径 vs 对应单卡路径 | 不同元素/24,576 | 相对 L2 |
| --- | --- | --- |
| 仅基座，无 LoRA | 9,770 | 0.00237586 |
| 标准低秩局部输出归约 | 8,818 | 0.00248986 |
| O 归约后 expand | 2,465 | 0.00135634 |
| FP32 局部累加及归约 | 17 | 0.0000398388 |

原生 GEMM 另存为对照：基座9,774个不同，FP32控制7个不同。BI降低或改变了部分误差，没有实现跨TP逐元素相同。同一真实输入的 Q 列分片在本轮 BI 单层对照中全部相等；这不代表任意输入或所有层都相等。

单层报告同时记录原生算子与 BI 算子结果；HCCL 运行必须核对 `backend=hccl`、`world_size=2` 及实际 BI 后端字段。`low_rank_precision.py` 中的两个局部 GEMM 加法只是模拟分片，不能当成 HCCL 实测。

因此跨 TP 不能只修 adapter 的索引或分片：基座投影已有不同的浮点求值路径。即使降低一个投影的误差，仍不能据此宣布整个 attention/MoE 模型跨 TP 的 token 全部一致。

## 5. 可复现工具

沿用 [资产和服务说明](reproduce.md)，启动默认标准低秩服务，不添加 `lora_execution_mode=merged`。复制 `diagnose.py`、`trace.py` 到服务目录的 `m31_lora_diagnose.py`、`m31_lora_trace.py`，将 `standard_controls.py` 也复制到工作目录，服务增加：

```bash
--worker-extension-cls standard_controls.StandardPrecisionProbe
```

服务 ready 后，在同一工作目录通过 `.minimax-lora-precision.json` 指定模式：

```json
{"mode":"post_reduce"}
```

分别运行 `original`、`post_reduce`、`fp32_local`，每次给原 `verify.py` 使用新的输出文件和 trace 目录。TP2/DP2 使用同拓扑 merged-A/B 参考及 `--expected-dp 2`；TP1/DP1 使用 TP1 merged 参考及 `--expected-dp 1`。对两套结果的同名 adapter、prompt 比较原生输入/输出 token ID 和 finish_reason。结束后删除本轮 precision marker。

单层实验使用此前在共同 token 前缀上采集的输入快照，仅本地保存 tensor，不上传权重：

```bash
python low_rank_precision.py \
  --weights /models/validation \
  --snapshot-dir /results/common-input \
  --adapter b --output /results/layer-b.json

HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1 \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tp_precision.py --weights /models/validation \
  --snapshot-dir /results/common-input \
  --adapter b --output /results/hccl.json
```

当前工具固定验证第一层 Q/O，模拟 TP2，并要求快照目录只有一个活动 DP 副本的 TP0/TP1 输入；不是完整模型验证器。控制扩展会读取 marker、转换权重/激活并增加计算，禁止作为生产或性能结果。

## 6. 当前交付边界

标准低秩原路径的双 adapter 加载、切片和 BI 服务隔离已验证；merged 模型硬指标和跨 TP 硬指标仍失败。本轮交付复现工具、实际对照和未解决原因，不将失败的精度实验包装成生产修复。全层、真实训练 adapter 和性能仍未验收。

若后续采用标准低秩的独立未合并参考及数值误差验收，需要明确数值阈值、参考实现和任务精度门槛；混合服务仍可保留逐 token 一致要求。本轮没有自行调整原规则，也没有把数值误差小等同于严格通过。

新诊断工具已实际执行单层 NPU、两卡 HCCL 和六组完整请求矩阵；格式检查通过 Ruff0.14.0 lint/format、codespell、Markdown lint及相关仓库静态检查。`bash format.sh ci` 再次尝试，因缺少 pre-commit 未执行完整 hooks，不报告完整CI通过。
