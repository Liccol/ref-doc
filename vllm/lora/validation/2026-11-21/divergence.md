# BF16 merged 与动态 LoRA 的 token 分歧定位

**结论：生产路径仍未通过模型 token 全一致验收。** 沿用同一四层 BF16 base、std0.1 的 A/B、四条 prompt 和32个生成 token，没有替换 fixture 或放宽比较条件。上游 vLLM 固定 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，Ascend 生产代码仍为 `9572a92cc33141bf023e1c89bcba95e4287130b7`。

## 对照结果

所有 LoRA 服务均为 TP2/DP2/EP、BF16、eager、batch-invariance；merged 参考关闭 LoRA。

| 执行路径 | 对同拓扑 merged 的 token 全一致 | 性质 |
| --- | --- | --- |
| 原始动态 LoRA，重新执行 | 3/8 | 生产模型验收失败；服务混合 vs 串行仍为36/36一致 |
| 保留 BF16 base GEMM，把低秩 kernel 换成独立 CPU FP64 双矩阵乘 | 3/8 | 数值定位；提高低秩计算精度没有消除模型验收失败 |
| 将目标层改用各 adapter 的 BF16 merged 权重做完整 GEMM | 8/8 | 因果对照，已经替换低秩执行方式，不能计作生产 LoRA 通过 |
| 上一行的完整 GEMM 对照，再与 TP1 merged 比较 | 4/8 | 另有跨 TP 计算顺序差异 |

低秩 FP64 对照与生产 LoRA 的完整32-token序列也仅4/8一致，尽管两者对 merged 的通过项相同。不能把“同为3/8”描述成输出完全相同，也不能据此声称 kernel 没有任何数值误差。

原始动态 LoRA 对同拓扑 merged 的失败位置（prompt 从0编号，token 位置从0编号）：

| Adapter | Prompt | 首个分歧位置 |
| --- | --- | --- |
| A | 0 | 17 |
| A | 1 | 22 |
| A | 2 | 5 |
| B | 0 | 4 |
| B | 2 | 0 |

## 逐层证据

1. 将每条失败用例分歧前的共同 token 前缀作为输入，两个模型都只生成下一个 token。五条用例仍复现相应分歧，排除了后续生成历史不同的影响。
2. 实际加载的全部 Q/K/V/O 基座分片和非零 adapter 槽位均与 checkpoint 的独立切片比较通过，覆盖两个 TP rank 和两个 DP 副本。独立检查 alpha/r、Q/K/V 输出切片、O 输入切片和 KV 复制规则，没有复用 wrapper 的切片辅助函数。
3. 对这些共同输入，生产 LoRA 与 `BF16 base output + FP32 low-rank delta` 独立参考的最大相对 L2 误差为 `7.305540202651173e-5`。这项误差指标用于定位，不替代 token 硬指标。
4. 第0层 QKV 的输入逐元素相等，indexer 尾部也逐元素相等；Q/K/V 输出的相对 L2 差异为约 `0.00265–0.00280`。首个数值差异已经出现在 attention 和 MoE 之前。
5. Adapter-B / prompt2 无需追加历史即可复现：第3层最后一个输入 token，按真实配置 `topk(sigmoid(gate_logits) + correction_bias)`，LoRA 的专家为 `[8,111,60,17]`，merged 为 `[8,111,60,74]`。其 logits 相对 L2 差异约 `0.11975`，最终 top1 分别为 token `152356`、`15039`。这是 MoE 离散路由放大前序误差的具体用例；不是所有失败都必然由专家集合变化解释。

完整 GEMM 控制的8/8仅指生成 token：该控制在 B/prompt2 的16组模块快照中，13组完全相等，第3层 MoE 输出仍有约0.00213相对 L2 差异，最终 logits 仍有约0.00222差异。不能把它扩大为所有内部 tensor 完全相等，或任意 prompt 的一致性保证。

## 为什么提高低秩精度仍不能保证一致

两条路径的舍入位置不同：

```text
merged:       BF16_GEMM(X, BF16(W + scale * B @ A))
dynamic LoRA: BF16(BF16_GEMM(X, W) + (X @ A.T) @ (scale * B).T)
```

数学上的乘法分配律不保证上述 BF16 运算等价。第0层 Q 的一个64×64子矩阵实测：理想更新的 FP64 秩为8，保存后的 `W_merged_BF16 - W_base_BF16` 的 FP64 秩为64；二者相对 L2 差异 `0.0018990490498036973`。合并舍入后的有效更新已经不再是原来的 rank8 更新，修改分片或 alpha 无法把它重新变成同一个 rank8 矩阵。

因此，当前固定用例的严格验收保持失败。标准动态低秩实现无法保证对任意 prompt 与 BF16 merged 得到相同 token。采用每个 adapter 的完整合并目标层做 GEMM，在同拓扑的本次用例中可以8/8一致，但会增加每个 adapter 的权重内存并改变执行方式，应作为单独方案评估，不能混入普通 LoRA 的验证结论。

## 复现定位

沿用 [启动配置](reproduce.md)，将 worker extension 替换为 `m31_lora_diagnose.LoRADiagnose`，并把 [diagnose.py](https://github.com/Liccol/vllm-ascend/blob/0bba169a1653b946ad9efe4e4855938ca08e75fa/tests/e2e/models/minimax_m31_lora/diagnose.py) 复制到服务工作目录的 `m31_lora_diagnose.py`。该扩展只在测试中使用，不修改 vLLM 或生产模型代码。

服务 ready 后，在共同工作目录写入 `.minimax-lora-diagnose.json`：

```json
{
  "role": "base",
  "label": "b-p2-prefix0",
  "directory": "/workspace/lora-debug",
  "weights": "/models/validation",
  "snapshots": true
}
```

`role` 必须等于服务的 `--served-model-name`；`weights` 下须有 `base`、`adapter-a`、`adapter-b`。逐条发送固定 token-ID 请求后移除 marker。扩展忽略 DP dummy 执行，按 DP/TP 分别保存逐层 tensor 和权重/kernel 检查 JSONL；不用于性能测试。

给 merged 服务捕获快照时，`role` 改成相应服务名，使用独立输出目录。两个测试控制通过 marker 的 `mode` 指定：

- `unmerged-reference`：保留原 BF16 base 输出，使用 CPU FP64 计算低秩增量；逐 rank 记录 `test_only_fp64_delta=true`。
- `merged-reference`：增加 `"adapter":"a"` 或 `"adapter":"b"`，为该 adapter 缓存完整合并目标层，仅支持串行单 adapter 请求；逐 rank 记录 `test_only_dense_merge=true`。

两个控制均实际覆盖8条请求，每条请求的8个目标模块及两个 TP rank，共128条控制记录。报告必须核对这些记录，不能只凭 marker 推断控制生效。公开附件只包含数值统计和请求结果，不包含 tensor 快照。

## 归档与检查

[诊断代码提交](https://github.com/Liccol/vllm-ascend/commit/0bba169a1653b946ad9efe4e4855938ca08e75fa)；[原始请求、逐层统计和逐 rank 控制证据](divergence-evidence.tar.gz)，共22个文件。附件 SHA256：`d0abf781b779b28a14ce6b87569e27573cb91adbd6fa3bb5e21d74c04382f387`。

Ruff0.14.0 lint/format、codespell2.4.1、markdownlint0.45.0、局部仓库 Python 检查和 `git diff --check` 通过。`bash format.sh ci` 因缺少 pre-commit 未完成，不将局部检查称为完整 CI。原19项 UT 的通过记录见前次报告；本轮没有修改生产推理代码，新增扩展通过上述真实 NPU 请求验证。
