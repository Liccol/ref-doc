# 2026-11-21 实测结果

**当前采用标准低秩：** [继续定位与跨 TP 实测](standard-low-rank.md) 复现模型3/8、服务36/36；两种精度改动均未通过完整规则，未合入生产路径。标准低秩 TP1与TP2对比为3/8，不能计入历史 merged 缓存模式的通过结果。

**后续进展：** 新增的可选 `merged` 模式已在同拓扑 TP2/DP2/EP 下通过模型8/8、混合服务36/36，详见 [正式实现与验收](merged-mode.md) 和 [代码提交 `b9db9e22e`](https://github.com/Liccol/vllm-ascend/commit/b9db9e22e6f2d0b682e022a371d6cc514907f8fe)。下文保留默认低秩模式的历史结果；跨 TP1→TP2 仍只有4/8一致，全层与训练 adapter 尚未验证。

- [测试方案](validation-plan.md)、[复现步骤](reproduce.md)。
- [开发代码 `9572a92cc`](https://github.com/Liccol/vllm-ascend/commit/9572a92cc33141bf023e1c89bcba95e4287130b7)，位于 `minimax-m31-lora` 分支，未创建上游 PR。
- [原始请求及批次记录](evidence.tar.gz)，[SHA256](SHA256SUMS)。压缩包保留初始弱 fixture、增强 fixture 默认模式、BI模式、同拓扑 merged 对照和独立 NPU 数值诊断；不包含模型权重、地址和账密。

**默认低秩模式状态：未通过用户要求的全部验收。** 五段投影适配已能加载和执行两个 adapter，服务隔离在 batch-invariance 模式下通过；merged 与动态 LoRA 的 token ID 全一致仍未通过。不将数值近似改称为该硬性规则通过。

后续 [逐层分歧定位与因果对照](divergence.md) 已完成：独立 FP64 低秩仍为3/8；完整 merged GEMM 测试控制在同拓扑达到8/8。默认低秩路径的严格模型验收结论保持失败；可选 merged 模式单列报告。

## 1. 版本与资产

| 项目 | 本次实际范围 |
| --- | --- |
| vLLM | 主仓配套 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，未改代码 |
| Ascend 基线 | `7a7ed073c481d5e3a2fc2194c5b879efb2d373c0` 加本分支 Python 修改 |
| 环境 | A3；PyTorch 2.10.0、torch-npu 2.10.0.post7、Transformers 5.19.0 |
| 权重 | 真实 BF16 checkpoint 的连续前4层，原宽度/head/expert 数保留；3层 dense MLP + 1层 routed MoE，均含 sparse attention/indexer |
| LoRA | Q/K/V/O，rank8、alpha16、A/B 标准差0.1，两个独立 seed；非训练 adapter |
| LoRA 并行 | TP2、DP2、EP；实际4个 EP rank |
| merged 参考 | 关闭 LoRA、逐请求执行；分别运行 TP1/DP1 和 TP2/DP2/EP |
| 请求 | 固定4条 prompt，使用相同输入 token IDs，temperature0、seed0、ignore_eos、生成32 token |
| 执行模式 | eager、无 prefix cache、无 SP/FlashComm、无 speculative/DSpark |

源码在容器中固定到了上述 commit；复用了镜像已有二进制扩展，没有重编全套 C++/AscendC 算子。镜像残留 `0.33.0` 元数据曾误触发 release-only Mamba 分支，测试使用已有 `VLLM_VERSION=0.33.0.dev0` 选择 main 兼容路径。这是运行环境条件，不是修改 vLLM 源码。

## 2. 回退与失败复现

旧版未提交改动已移出工作树并归档，四个 tracked 文件恢复到基线。新实现只增加五段 QKV/indexer 层的精确 wrapper 匹配，复用上游 Q/K/V 分片和 slot 生命周期，不保留旧的 `max_loras=1` 限制。

未接入新 wrapper 时，在真实权重 TP2/DP2/EP 启动过程中复现：

```text
ValueError: LoRA target module ...layers.0.self_attn.qkv_proj
(AscendMiniMaxM3QKVParallelLinearWithIndexer) matched the deployment
configuration but could not be wrapped by any LoRA layer implementation.
```

修复后服务成功启动，并真实加载 A/B；后续所有比较均直接使用 API 返回的 token ID。

## 3. 判定结果

| 检查 | 实际结果 | 判定 |
| --- | --- | --- |
| 仓库 UT | 19/19，通过五段切片、TP/KV复制、多 slot、partial reload、bias、替换范围和注册回归 | 通过 |
| 初始小增量 fixture | std0.01，虽然 logits 改变，但 base/A/B 的生成 token 全部相同 | 区分能力不足，不接受为最终隔离证据 |
| 增强 fixture 区分能力 | std0.1；base↔A、base↔B、A↔B 各4/4 prompt 的 token 序列不同 | 有效 |
| 默认模式混合请求 vs 串行 | 36次比较中14次分歧，base请求12/12一致 | 服务规则未通过 |
| batch-invariance 模式混合 vs 串行 | 36/36一致，共1152个生成 token；含12条串行参考和三轮交错混合 | 服务规则通过 |
| BI：TP1 merged vs TP2/DP2/EP LoRA | A：1/4一致，B：2/4一致，总3/8一致 | 模型规则未通过 |
| BI：同拓扑 merged vs LoRA | A：1/4一致，B：2/4一致，总3/8一致 | 模型规则未通过 |
| BI：merged TP1 vs merged TP2 | A、B各2/4一致，总4/8一致 | 存在独立于 LoRA 的跨 TP 差异 |

BI 指 `VLLM_BATCH_INVARIANT=1`，同时明确关闭 chunked prefill，block size128。日志确认4个 LoRA workers 都进入该模式；没有仅根据开关推断生效。

有效 BI 混批证据：

| DP rank | TP rank | 元数据更新记录 | 含 base、A、B 的同 batch 记录 |
| --- | --- | --- | --- |
| 0 | 0 | 238 | 8 |
| 0 | 1 | 238 | 8 |
| 1 | 0 | 241 | 11 |
| 1 | 1 | 241 | 11 |

两个 TP rank 属于同一个 DP batch，表中的8/11不能按 TP rank 相加宣称独立 batch 数。trace 从 ready 后开始，排除了 dummy warmup。

## 4. 数值定位

独立运行真实 NPU Punica 低秩分支，6个 token 混用 base/A/B，对照逐 token 执行：

- 混合与逐条结果逐元素完全一致，最大绝对误差0。
- base 两行严格为0增量。
- 与独立 CPU FP32 低秩公式的相对 L2 误差 `2.2620342861046083e-5`。

再使用真实第0层 Q 投影的 TP2 首个 shard、真实 adapter-A 和保存的 merged-A，给定完全相同的随机 BF16 输入：

- 按记录公式重建的 merged 权重与保存权重逐元素一致，排除了这份 fixture 的合并公式/存盘不一致。
- 动态 LoRA 与 `BF16 base output + FP32 low-rank delta` 参考的相对 L2 误差 `1.2414776392688509e-5`。
- 动态 LoRA 与 BF16 merged GEMM 不逐元素相等，相对 L2 误差 `0.0027918571140617132`。

这证实相同输入的单层已存在两种计算顺序的 BF16 数值差异，并支持“数值路径差异会影响严格 token 对照”的判断。它不等于已经证明每处整模型分歧的唯一根因，更不构成豁免模型验收失败。

复现诊断：

```bash
uv run --no-project python kernel_check.py \
  --weights /workspace/lora-validation/weights \
  --output /workspace/lora-validation/kernel-real-check.json
```

## 5. 范围与后续门槛

- 当前只证明四层真实 BF16 attention adapter 的加载/执行和 BI 服务隔离；不宣布 M3.1 LoRA 已完整验收。
- 全60层、真实训练 adapter、TP8 的实际 KV-head 复制、专家/MLP/视觉 adapter、图模式和长上下文 sparse block 选择未验证。1024上下文不足以覆盖超过16个128-token block的稀疏选择边界。
- 保留用户“token ID 完全一致”的模型硬指标为失败。后续应在共同输入下逐层记录 hidden state/logits，继续界定 merged/BF16 基线与动态计算误差；如果需要调整验收口径，应由规则拥有者明确决定，不能靠测试脚本放宽。
- 修改的 Python 通过固定 Ruff0.14.0 lint/format，相关 Markdown 通过 markdownlint0.45.0，codespell及 `git diff --check` 通过。
- `bash format.sh ci` 已尝试，因容器缺少 pre-commit 未执行全套 hooks；不将上述局部检查称为完整 CI。
