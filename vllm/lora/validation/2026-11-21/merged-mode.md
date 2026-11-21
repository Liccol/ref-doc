# M3.1 BF16 LoRA merged 模式验收

本轮新增显式 `lora_execution_mode="merged"`，在同拓扑 TP2/DP2/EP 下通过模型8/8和服务36/36。默认 `low_rank` 路径仍为模型3/8、BI服务36/36；没有把低秩误差门槛放宽，也没有替换测试资产。

实现已推送到个人仓库 [`minimax-m31-lora` 分支的 `b9db9e22e`](https://github.com/Liccol/vllm-ascend/commit/b9db9e22e6f2d0b682e022a371d6cc514907f8fe)。vLLM 保持配套提交不变，未创建上游 PR。

[原始请求、参考输出、混批映射和启动日志摘录](merged-mode-evidence.tar.gz)，[SHA256](SHA256SUMS)。压缩包包含15个文件，无模型 tensor、内网地址或凭据；`summary.json` 记录六个源码文件按 Git 换行规范计算的哈希，便于核对实际测试版本。Ascend 父提交是既有 `0bba169a1`，本轮新增文件及修改另以源码哈希标识。

## 1. 原因与实现

默认低秩计算先得到 BF16 的基座输出，再加低秩增量；BF16 merged checkpoint 先对更新后的权重舍入，再做完整 GEMM。此前 [逐层诊断](divergence.md) 已确认，提高低秩计算到 FP64 仍不能通过模型硬指标；完整 merged GEMM 对照能在同拓扑达到8/8。

正式实现沿用 vLLM 的 adapter 加载、scaling、TP 分片、LRU slot 和 token 映射，在 adapter 激活时按下面的公式构建本地目标权重：

```text
merged = BF16(FP32(base) + FP32(scaled_B) @ FP32(A))
```

- 仅替换 M3 attention 的 QKV 与 O wrapper；五段 QKV 的 indexer 尾部保留基座值。
- 在 profiling 前为 `max_loras` 个 slot 预留完整目标层缓存，避免后续激活侵占已规划的 KV cache。
- reset 使缓存失效；重载从原始基座重新构建，partial adapter 不会继承旧 slot 的增量。
- dummy DP 的零 adapter 直接走基座，避免空闲副本反复做 CPU 合并。
- forward 计算基座及每个启用 slot 的完整投影，用设备侧 token→slot 映射选择输出；不增加逐 token CPU 同步。

这是额外显存与计算换取另一种 BF16 求值顺序的模式，不能称为低秩 kernel 精度修复或性能优化。

## 2. 版本与测试资产

vLLM 固定主仓配套提交 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，未修改其源码。Ascend 基线为 `7a7ed073c481d5e3a2fc2194c5b879efb2d373c0`；本轮在既有适配与诊断提交上追加实现，未升级 Transformers。

环境为 A3、PyTorch 2.10.0、torch-npu 2.10.0.post7、Transformers 5.19.0，复用镜像已有二进制扩展。版本元数据兼容条件仍为已有的 `VLLM_VERSION=0.33.0.dev0`，不表示依赖升级。

沿用真实 BF16 checkpoint 连续前4层：3层 dense MLP、1层 routed MoE，宽度6144、Q64/KV4、128专家/top-k4不变。A/B 为独立 seed 的非零合成 Q/K/V/O adapter，rank8、alpha16、标准差0.1。固定4条 prompt，每条32个生成 token，temperature0、seed0、ignore_eos；逐项比较原生输入/输出 token ID、长度与 finish_reason。

参考和待测服务均启用 batch invariance、eager、TP2/DP2/EP，关闭 prefix cache、chunked prefill、SP/FlashComm 和 speculative；block size128、最大上下文1024。merged 参考关闭 LoRA，逐请求采集；服务对照包含12条 base/A/B 串行请求及三轮交错并发请求。

## 3. 正式实现结果

正式服务使用普通 `LoRABatchTrace`，没有加载原型或诊断算术补丁。

| 检查 | 结果 |
| --- | --- |
| 同拓扑 merged-A vs base+adapter-A | 4/4完全一致 |
| 同拓扑 merged-B vs base+adapter-B | 4/4完全一致 |
| base/A/B 混合请求 vs 各自串行请求 | 36/36完全一致，共1152个生成 token |
| base↔A、base↔B、A↔B 区分能力 | 每组4/4 prompt 的生成序列不同 |
| LoRA 与 Ascend 配置目录回归 | 301 passed，17 warnings |
| TP1 merged vs TP2/DP2/EP merged 模式 | 4/8一致，仍未通过跨 TP 全一致 |

真实 batch trace 从 ready 后开始，排除 warmup：

| DP rank | TP rank | mapping 更新 | 同 batch 含 base/A/B |
| --- | --- | --- | --- |
| 0 | 0 | 239 | 4 |
| 0 | 1 | 239 | 4 |
| 1 | 0 | 239 | 6 |
| 1 | 1 | 239 | 6 |

同一 DP 的两个 TP rank 是同一个 batch，不能将它们相加宣称独立 batch 数。

正式实现独立重启复跑两次均通过8/8和36/36。表中为最终源码的第二次记录；首轮两个 DP 各有7和8次真正混批。批次调度数量可以不同，每条请求的一致性判定没有改变。

跨 TP 的剩余差异必须保留：此前关闭 LoRA 的 merged TP1 vs merged TP2 本身也只有4/8一致。如果“单推”要求 TP1 单设备参考，本轮仍未满足这一要求；8/8只适用于相同并行拓扑下的逐请求参考。

## 4. 开销与兼容范围

启动日志确认每个 worker 创建4个 merged QKV wrapper 和4个 merged O wrapper；两个 slot 的逻辑缓存总计893,386,752字节，即0.83203125 GiB/worker，四个 worker 合计3.328125 GiB。该值仅为目标权重缓存，不含原始权重、A/B、临时合并缓冲、KV cache 或分配器开销。

每个 worker 的逻辑缓存公式为 `max_loras × Σ(本地目标权重元素数 × 2)`。QKV 缓存包含未修改的 indexer 尾部。全层估算必须按实际逐层结构与 TP 分片重新求和，不能直接把减层测试的显存当作全模型容量。

| 场景 | 当前结论 |
| --- | --- |
| BF16 attention Q/K/V/O、eager、TP2/DP2/EP | 本轮验证通过 |
| TP/KV复制、bias、2D/3D、partial reload、slot reset/复用 | CPU UT覆盖；不等同于所有拓扑NPU实测 |
| fully-sharded、SP/FlashComm1、图模式、sleep、基座权重切换 | 配置拒绝；缓存或 token 映射生命周期尚未适配 |
| expert/MLP/indexer/视觉 adapter | 不在允许的 target modules 中 |
| 全60层、训练 adapter、视觉请求、MTP/DSpark、长上下文 | 未验证；本轮为文本四层正确性定位，1024上下文不覆盖128k容量与长稀疏边界 |
| 吞吐、TTFT、TPOT、加载耗时与高频切换 | 未做性能验收；trace包含文件IO，不作为性能数据 |

## 5. 复现

按 [测试说明](reproduce.md) 准备同一份资产、同拓扑 merged-A/B 参考，并在 LoRA 服务命令中增加：

```bash
--additional-config '{"lora_execution_mode":"merged"}'
```

同时保持 `--dtype bfloat16 --lora-dtype bfloat16 --enforce-eager`、显式 `--lora-target-modules q_proj k_proj v_proj o_proj`、`max_loras=2` 和全部 BI 设置不变。参考与 LoRA 服务都应设置 `VLLM_BATCH_INVARIANT=1`，并使用 `--no-enable-chunked-prefill --block-size 128`。

使用原 `verify.py`，给 `--reference-a/--reference-b` 传入 TP2/DP2/EP 的 merged 参考，设置新的 `--trace-dir`、`--expected-dp 2` 和输出文件。不要修改 prompt、adapter 强度、token 长度或一致性判定。TP1参考另存另比。

相关回归命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_VERSION=0.33.0.dev0 \
python -m pytest -q tests/ut/lora tests/ut/test_ascend_config.py
```

修改文件通过 Ruff0.14.0 lint/format、codespell、Markdown lint、相关仓库静态检查和 `git diff --check`。`bash format.sh ci` 已尝试，因环境缺少 pre-commit 未运行完整 hooks；这些局部检查不等于完整 CI。无上游 PR，本轮只推送个人分支；发布说明片段及 PR 编号在准备上游 PR 时补齐。
