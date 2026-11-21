# MiniMax-M3.1-Flash BF16 LoRA 验收

验收标准和减层边界见 [开发与测试方案](validation-plan.md)。脚本来自 [固定开发提交](https://github.com/Liccol/vllm-ascend/tree/9572a92cc33141bf023e1c89bcba95e4287130b7/tests/e2e/models/minimax_m31_lora)，在该目录读取命令，再按文中要求从服务工作目录运行。

本目录验证 attention Q/K/V/O adapter。构造的 A/B 是两套独立、非零、固定 seed 的合成 PEFT 权重，初始化标准差为0.1，用于验证计算和服务链路，不代表经过微调的任务质量。

本轮 [实测结果](README.md)：BI服务隔离通过，merged模型token全一致未通过；不能将本分支称为完整验收通过。

## 1. 资产准备

固定 vLLM `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15` 与 Ascend 基线 `7a7ed073c481d5e3a2fc2194c5b879efb2d373c0`，安装本分支代码。确认 import 路径与预期源码一致。

```bash
uv run --no-project python prepare.py \
  --source /models/MiniMax-M3.1-Flash-BF16 \
  --output /workspace/lora-validation/weights \
  --layers 4 --rank 8 --alpha 16
```

输出 `base`、`adapter-a`、`adapter-b`、`merged-a`、`merged-b` 和 `manifest.json`。工具拒绝覆盖现有输出，拒绝输入配置中的量化标记和非 BF16 attention tensor。未修改的 merged shard 通过 symlink 复用 base，因此移动资产时必须保留目录关系。

源 checkpoint 必须有完整 safetensors index，文本层名为 `language_model.model.layers.N`。工具裁剪连续前 N 层及对应配置数组，保留 embedding/final norm/LM head/视觉权重；实际 EP 覆盖还须核对保留层包含 routed MoE。

## 2. 服务命令

从 `/workspace` 运行。以下示例单机四设备 TP2、DP2、EP；设备选择及通信网络按测试机配置。确认设备空闲、端口空闲，不停止其他任务进程。

```bash
uv run --no-project python -m vllm.entrypoints.cli.main serve \
  /workspace/lora-validation/weights/base \
  --served-model-name base --host 127.0.0.1 --port 8000 \
  --api-server-count 1 --dtype bfloat16 \
  --tensor-parallel-size 2 --data-parallel-size 2 \
  --enable-expert-parallel --distributed-executor-backend mp \
  --language-model-only --enforce-eager --no-enable-prefix-caching \
  --max-model-len 1024 --max-num-batched-tokens 1024 --max-num-seqs 16 \
  --gpu-memory-utilization 0.6 --generation-config vllm
```

先关闭 LoRA，分别将模型路径/服务名替换为 `base`、`merged-a`、`merged-b`，对各自服务采集单请求参考。可以顺序运行同一个端口，也可在独立设备组上并行启动不同端口；必须记录各自拓扑。

```bash
uv run --no-project python verify.py \
  --url http://127.0.0.1:8000 \
  --tokenizer /workspace/lora-validation/weights/base \
  --models merged-a --rounds 0 \
  --output /workspace/lora-validation/merged-a.json
```

上例请求的 model 名必须与对应服务一致。采集 base/merged-b 时同步修改 `--models` 和输出文件。采集完成只证明参考请求执行成功，最终模型验收还须比较 LoRA 输出。

## 3. LoRA 混合服务

将 `trace.py` 复制到 `/workspace/m31_lora_trace.py`。它通过上游 worker 生命周期扩展在每个 DP/TP worker 记录真实 Punica token mapping；仅供正确性测试，会增加文件 IO，禁止用于性能数据。

在第2节 base 服务命令中增加：

```bash
  --enable-lora --max-loras 2 --max-cpu-loras 2 \
  --max-lora-rank 8 --lora-dtype bfloat16 \
  --lora-target-modules q_proj k_proj v_proj o_proj \
  --lora-modules \
    lora-a=/workspace/lora-validation/weights/adapter-a \
    lora-b=/workspace/lora-validation/weights/adapter-b \
  --worker-extension-cls m31_lora_trace.LoRABatchTrace
```

确认 `/v1/models` 返回 base、lora-a、lora-b 后，从相同工作目录和文件系统运行：

```bash
uv run --no-project python verify.py \
  --tokenizer /workspace/lora-validation/weights/base \
  --reference-base /workspace/lora-validation/base.json \
  --reference-a /workspace/lora-validation/merged-a.json \
  --reference-b /workspace/lora-validation/merged-b.json \
  --trace-dir /workspace/lora-validation/batches \
  --expected-dp 2 \
  --output /workspace/lora-validation/result.json
```

工具先逐条请求 base/A/B，再执行三轮并发交错请求，4种 prompt 长度、每次32个生成 token。直接记录 API 原生 `token_ids`、输入 token ID、finish_reason 和 top-k logprobs，不使用重新 tokenize 的解码文本。任何 token/长度/finish_reason 不一致均返回非零。

trace 从 ready 后的客户端 marker 开始，排除 dummy warmup。还检查实际覆盖预期 DP rank，并要求已观察到的 workers 记录含 base/adapter1/adapter2 的同 batch token 映射。缺少证据也判失败，不能以并发 HTTP 200 代替实际混批。

严格服务一致性应同时验证现有 batch-invariance 模式：在参考和 LoRA 服务启动前设置 `VLLM_BATCH_INVARIANT=1`，并增加 `--no-enable-chunked-prefill --block-size 128`。本轮该模式消除了服务混批分歧，但没有消除 merged 模型对照分歧；不能把模式开关当作模型验收保证。

运行前确保没有旧的 `.minimax-lora-trace.json` marker，trace 输出目录必须不存在。客户端正常或异常退出都会清理本轮 marker；若客户端被强杀，应手动删除该单个 marker 后再重跑。

## 4. 判定与扩展

- 先检查 `result.json.failures`，再分别核对 merged 对照、服务混批对照和 batch trace。报告首个 token 分歧，不将 BF16 舍入解释当作通过条件。
- 检查 adapter A/B 文件哈希不同、A/B 非零，并核对 LoRA logits 相比 base 的变化，排除未生效假通过。
- base↔A、base↔B、A↔B 每组至少一个 prompt 的生成 token 必须不同，否则工具判为测试 fixture 缺少区分能力，不能仅凭三者输出相同宣称隔离通过。
- 首轮关闭 SP/FlashComm、speculative/DSpark、图模式；fully-sharded 和专家/视觉 adapter 不在本轮覆盖范围。
- 减层通过仅证明该减层结构的链路。增加层数、全层、真实训练 adapter 和其他并行拓扑分别记录结果，不继承“通过”结论。
- 镜像版本元数据与源码可能不同。本次固定 main 源码的测试环境曾因残留 `0.33.0` 标识触发 release-only Mamba 分支；使用已有 `VLLM_VERSION=0.33.0.dev0` 选择 main 兼容路径后运行。正式环境应正确安装对应版本，不靠浮动升级绕过错误。
