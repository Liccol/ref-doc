# MiniMax-M3.1-Flash BF16 LoRA 开发与验收方案

**Goal:** 按 merged 对照和多 adapter 混合服务两条规则，开发并验证 M3.1 BF16 attention LoRA。

**Architecture:** vLLM 固定主仓配套 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，Ascend 基线 `7a7ed073c481d5e3a2fc2194c5b879efb2d373c0`。复用上游 loader、manager、TP 分片和 NPU Punica，为五段投影增加精确匹配的 wrapper；不沿用旧版单 adapter 限制。

**Tech Stack:** PyTorch BF16、vLLM、vLLM Ascend、PEFT safetensors、pytest、OpenAI completion API。

## 1. 硬性验收规则

| 编号 | 对照 | 通过条件 |
| --- | --- | --- |
| M1 | 真实 BF16 base + adapter A，与从同一 base、同一 A 生成的 merged-A 单请求推理 | 每个固定 prompt 的生成 token ID、长度、finish_reason 一致 |
| M2 | 同上，换为独立 adapter B / merged-B | 同上；A/B 文件哈希不同，A/B 非零且确实加载 |
| S1 | base、A、B 各自串行请求，与三者并发交错服务 | 对每个 model/prompt，生成 token ID、长度、finish_reason 一致 |
| S2 | 不同 prompt 长度、prefill/decode 交错、请求顺序变化、多轮 A→B→base→A | 不串 adapter、无残留；每轮与相应串行参考完全一致 |
| P1 | LoRA 服务启用 TP>1、DP>1、EP | 各 DP rank 实际承载请求；两个 adapter 实际在同一执行 batch 中出现的证据；不能只检查启动参数 |

模型正确性：merged 参考关闭 LoRA，LoRA 侧启用 TP+DP+EP；优先让参考侧使用相同并行与 attention 配置、单请求执行，隔离 LoRA 差异。资源允许时补 TP1 单卡参考，单列跨并行差异。不能把“单请求”误当“必须单卡”。

服务正确性：单独请求参考保持 `max_loras=2` 的同一服务配置，与混合请求比较；额外比较 `max_loras=1` 专用路径，区分不同 kernel。base 另与完全关闭 LoRA 的基座服务比较。

精确 token 一致是硬指标。BF16 中 `X(W+BA)` 与 `XW+(XA)B` 的舍入、batch 大小和通信归约顺序可能不同；出现分歧仍判该项失败，记录首个分歧位置和固定共同前缀下的 logits/top-k，定位原因后重测，不自动按容差放行。

## 2. 固定条件与有效性检查

- 固定 tokenizer/chat template 版本及实际输入 token ID；首轮用 `/v1/completions` 的 token-ID prompt 消除模板差异。
- `temperature=0`、`top_p=1`、`seed=0`、`n=1`，固定 max_tokens、EOS/stop、penalty、logits processor。不以“seed 相同”推断跨 batch 确定性。
- 请求 `return_token_ids=true`，直接比较 API 返回 token ID，禁止把解码文本重新 tokenize 作为原始输出证据。
- 首轮 eager，关闭 prefix cache、SP/FlashComm、speculative/DSpark。通过后补实际所需 graph 配置，不混用不同执行模式结果。
- 校验 adapter key 消费、目标层 wrapper、rank/alpha、权重分片、槽位及 token→adapter 映射。两个 adapter 不可只是同一文件换两个 ID。
- 零 adapter 只做基座等价控制。除了确认非零权重和 logits 变化，测试集必须包含可区分 base/A/B 的生成 token：base↔A、base↔B、A↔B 每组至少一个 prompt 不同，否则标为 fixture 覆盖不足，不接受为服务隔离通过。
- TP2/DP2/EP 为首个四设备拓扑，之后 TP4/DP2/EP、TP8/DP2/EP。EP group 的实际大小和专家分布以日志为准，不额外乘一个 EP 倍数计算卡数。
- 同时保留客户端请求结果和逐 rank 执行证据。DP 调度可能把 A/B 分到不同副本，单纯并发请求不足以证明 multi-adapter batch；必要时先 DP1 强制混批，再在 DP2 配置复现同副本混批。

## 3. 减层权重方案

可以用于开发和以上规则的减层模型验收，但不能据此声称全层原模型已通过。优先保留连续前 N 层，覆盖至少一层真实 sparse attention/indexer 和 routed MoE；不能仅留 dense MLP 而空跑 EP。

1. 基于真实未量化 BF16 checkpoint，检查 tensor dtype/shape，不仅看 `torch_dtype`。W8A8/NVFP4 不能靠改 dtype 字段变成 BF16。
2. 同步裁剪 `text_config.num_hidden_layers`、`layer_types`、`mlp_layer_types` 及所有实际按层配置；保留 hidden/head/expert/top-k 等宽度与路由参数，保留 embedding、final norm、LM head。
3. 删除被裁层的权重，重建 safetensors index；保留 shard 时只读取有效 key，记录实际资产大小。不得改写源 checkpoint。
4. 若保留非连续层，必须显式记录 old→new 层号映射，同时映射基座、adapter 和配置。本轮优先避免这种额外变量。
5. 从同一减层 base 构造 A/B PEFT adapter，按真实 Q/K/V/O shape、固定不同 seed、非零 A/B，`rank=8, alpha=16`、A/B 初始化标准差 `0.1`；每个 adapter 覆盖保留 attention 层。它们是合成计算 fixture，不是训练效果证据。
6. 独立按 `W_merged = (W.float() + (alpha/r) * B.float() @ A.float()).to(bfloat16)` 生成 merged-A/B；alpha 只应用一次，记录合并精度。禁止修改后又给 merged 模型挂同一个 adapter。
7. 保存源版本、保留层号/类型、config/tokenizer/adapter 哈希、tensor 清单、生成 seed 和 merge 公式。base/merged-A/merged-B 的 tokenizer 必须一致。

4层→8层→全层依次扩大。若前4层未覆盖 sparse MoE，则选择满足覆盖条件的最小连续前缀。减层会改变输出语义；只比较同一个减层模型的两个实现，不与原全层文本对答案。

## 4. 开发任务

### Task 1：回退和建立基线

- 已将旧四个 tracked 修改恢复到 HEAD；旧新增 wrapper/UT/计划移至仓库外归档。
- 核对目标容器实际 import 路径和两个 commit。使用容器内隔离工作目录/checkout，不改其他任务容器。
- 检索两个仓库的 adapter fixture；现有其他架构 LoRA 不可直接用于 M3.1。

### Task 2：五段投影包装

**Files:** `vllm_ascend/lora/utils.py`，`tests/ut/lora/test_utils.py`。

- 测试先复现自定义五段层无法替换及 slice 数错误。
- 为 `AscendMiniMaxM3QKVParallelLinearWithIndexer` 注册精确类型 wrapper，使用普通三段 QKV 的 A/B 和已有 TP/KV-replica 分片规则，仅更新输出前缀。
- 保留 base forward、bias、五段返回 shape；给 Punica 连续 QKV tensor，写回三段，不改 indexer 尾部。
- 不限制 `max_loras=1`，复用现有多槽位 Punica；不全局放宽普通 wrapper 的类型匹配。
- UT 验证 TP 分片/复制、base/A/B token 混合、部分目标、reset/重载、尾部与 bias；用真实 torch 参考计算。UT 不代替 NPU 验收。

### Task 3：真实 fixture 和服务验收工具

**Files:** `tests/e2e/models/minimax_m31_lora/prepare.py`、`verify.py`、中文 README。

- prepare 工具流式处理真实 safetensors，生成减层 base、独立 A/B 和 merged-A/B 清单；显式拒绝不支持的量化输入或不符合预期的 key/shape。
- verify 工具采集四种服务（base/merged-A/merged-B/LoRA）的原始 token ID，跑串行/混合/多轮用例，输出逐请求 JSON 和首个分歧，任一不一致退出非零。
- 复用现有请求/模型入口；混合客户端使用并发 barrier 发起请求。实际同 batch 与 DP 覆盖仍由设备侧 trace 证据补足。

### Task 4：设备验证与结果归档

- 先运行 UT 和 NPU 投影数值测试，再启动减层真实权重服务，执行 M1/M2/S1/S2/P1。
- 任何错误保留首个失败与修复后的重跑记录；无法启动时写明未执行的后续项，不将 HTTP ready 视为精度通过。
- 覆盖门槛通过后再试全层；缺资产/资源则明确标记 pending，不能将 dummy 或合成小宽度模型冒充真实 M3.1。
- 完成后一个 signed commit 交付；PR 编号确定后再按仓库要求补 release note。测试报告仅含脱敏路径和通用拓扑，不包含账密或内部地址。
