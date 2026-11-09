# 全层 BF16 开箱测试记录与复现

> 环境脱敏说明：本文及关联命令、JSON、压缩结果中的地址、容器名、账号和内部路径已替换为示例。示例不可直接作为部署配置；测试数值及结论保持不变。SHA256SUMS 对应脱敏后的文件。


完整逐用例检查项、逐轮比较与显存实测表见 [验证详单](VALIDATION_DETAILS.md)。

## 环境与版本

双机 Ascend A5，各 8 张物理卡，容器 `example-container`。完整权重路径 `/path/to/artifact`，60 层语言模型；历史权重文件共 204 分片、23428 张量，文件总量约 795.536 GiB。

2026-10-26 初始环境记录：vLLM-Ascend commit `5666e6616ecc0155fb9ae79d861eb9f9781e6f31`，配套 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。这是初始记录，不保证后来所有独立复测运行实例的源码均未变化；复现前应核对实际加载路径和 commit。NumPy 1.26.4、PyArrow 20.0.0；CANN 根目录 `/path/to/installation`，没有足够证据统一声明所有轮次的 CANN build 号。

## 服务启动

进入各自容器，把本目录及 `workflow` 保持相对布局放入可访问路径。在 25 机执行 `bash full/commands/serve.sh 0 baseline`，26 机执行 `bash full/commands/serve.sh 1 baseline`。脚本包含完整原生命令：TP4、DP4、每机 DP2、EP、BF16、FULL_DECODE_ONLY、context 131072、max-num-seqs 16、batched tokens 8192、memory utilization 0.85；未启用投机解码和 dspark。两机使用 example_if，服务端口 31070，DP RPC 31071。

```bash
# 两台机器分别执行；启动顺序为 head → worker
docker exec -it example-container bash
# 25 机
bash full/commands/serve.sh 0 baseline
# 26 机另一个终端
bash full/commands/serve.sh 1 baseline
```

Sleep / checkpoint 恢复测试将两台命令中的 `baseline` 改为 `sleep`，额外启用 dev mode、sleep mode、weight_nz_mode=0 并关闭 prefix cache。SP 改为 `sp`，额外设置 `enable_flashcomm1=true`。PP 改为 `pp`，使用 TP4 PP4 DP1、原生双机 node-rank、master 192.0.2.10:29502。

以上 Bash 入口是依据历史参数整理的等价原生命令，没有在新目录重跑；历史包装器命令与各用例附加参数见 [historical-commands.txt](commands/historical-commands.txt)。机器 IP、网卡与权重挂载是当时环境，复用时须按目标环境配置。

## 测试命令与结果

### 文本双接口

服务就绪后在 head 容器执行：

```bash
curl -fsS http://127.0.0.1:31070/health
curl -fsS http://127.0.0.1:31070/v1/models
curl -sS http://127.0.0.1:31070/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash-BF16","prompt":"请计算1+1。","max_tokens":256,"temperature":0,"seed":0}'
curl -sS http://127.0.0.1:31070/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash-BF16","messages":[{"role":"user","content":"请计算1+1。"}],"max_tokens":256,"temperature":0,"seed":0}'
```

上述 curl 为人工复现 smoke 请求，不冒充历史逐 DP 验收请求。历史文本与真实图文基线通过，见 [baseline.json](evidence/baseline.json)。历史路由、SP、PP 的双接口/图文测试命令及原始脚本路径已收录在命令归档，结果见 [路由](evidence/routing_replay.json)、[路由层覆盖](evidence/routing-layer-completeness.json)、[SP](evidence/sp.json)、[PP](evidence/pp.json)。路由只验证捕获，不包含训练端 replay。

### Sleep / 同 checkpoint 恢复

`commands/sleep_determinism_probe.py` 和 `workflow/rl` 保存了原测试 Python 依赖；其中相对导入路径为此次移植所作调整，尚未重新上机验证。完整参数仍见命令归档。

```bash
# head 容器；从 minimax-m31-flash 根目录运行，先启用 sleep 服务
mkdir -p results
probe=full/commands/sleep_determinism_probe.py
report=results/level2.json
python3 "$probe" --action capture --level 2 --label control-0 --report "$report"
python3 "$probe" --action capture --level 2 --label control-1 --report "$report"
python3 "$probe" --action capture --level 2 --label control-2 --report "$report"
for i in 1 2 3; do
  python3 "$probe" --action capture --level 2 --label cycle${i}-before --report "$report"
  python3 "$probe" --action sleep --level 2 --label cycle${i}-sleep --report "$report"
  python3 "$probe" --action wake-weights --level 2 --label cycle${i}-wake-weights --report "$report"
  python3 "$probe" --action reload --level 2 --label cycle${i}-reload --report "$report"
  python3 "$probe" --action wake-kv --level 2 --label cycle${i}-wake-kv --report "$report"
  python3 "$probe" --action capture --level 2 --label cycle${i}-after --report "$report"
done
```

原验收还包含恢复后重复请求、逐 DP 比较和 HBM 检查，以上是生命周期复现入口，不能仅凭脚本跑完替代所有原验收检查。Level 1 将 level 改为 1，sleep 后按 weights、kv_cache 唤醒，**不执行 checkpoint reload**；原测试在 sleep 时失败，没有完成后续步骤。

| 用例 | 结果 | 证据 |
| --- | --- | --- |
| Level 2 三轮恢复 | 生命周期/HBM 通过；恢复前后 60/60 text/token/logprob 精确相等 | [原始 JSON.gz](evidence/sleep_level2_determinism_prefix_off.json.gz) |
| 同 checkpoint 重载 | 与 Level 2 共用三轮恢复数据；不是独立 awake 热更新，也不是不同权重切换 | [原始 JSON.gz](evidence/checkpoint_reload_per_dp_prefix_off.json.gz) |
| Level 1 | CPU pinned memory 分配报 aclrtMallocHostWithCfg/207001，HTTP500；wake 未验收 | [原始 JSON.gz](evidence/sleep_level1_determinism_prefix_off.json.gz) |
| IPC 更新 | clone 报 NPU OOM，更新和更新后输出验收未完成 | [原始结果](evidence/npu_ipc_weight_update.json) |
| HCCL 更新 | 缺独立 producer 卡，未执行完整事务 | [阻塞原因](evidence/hccl_weight_update.json) |
| DCP / PCP | 后端明确不支持，未绕过限制测试 | [DCP](evidence/dcp.json)、[PCP](evidence/pcp.json) |

IPC 的原 producer/consumer 调用和参数见历史命令归档，权重更新实现和命令入口在 `workflow/rl/weight-update/`。HCCL 没有已执行成功的测试命令，不能将参考命令标作实测。图文首次请求差异在 RL 操作前已出现：基线 16/20 精确一致，差异来自 4 个 DP 的图文请求；恢复前后则为 20/20。原始严格确定性失败与恢复通过的判定范围见 [逐 DP 报告](prefix-off-determinism-report.md)。

### 全层 AIME2026 三轮

基线服务同上。AISBench 采用 `vllm_api_stream_chat`，数据集 `aime2026_gen_0_shot_chat_prompt`，batch size 128、temperature 0、seed 1234、max_out_len 122880。

```bash
# 在装有 ais_bench 与数据集的 head 容器执行；config 路径需按安装位置确认
python3 full/commands/configure_accuracy.py \
  /path/to/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py \
  MiniMax-M3.1-Flash-BF16 31070 128 0 1234 122880
# 从评测工作目录执行，每轮保留新输出目录
for round in 1 2 3; do
  ais_bench --models vllm_api_stream_chat --datasets aime2026_gen_0_shot_chat_prompt
done
```

配置修改脚本从当时 accuracy skill 提取；命令中的 `/path/to` 是明确的环境占位符。原工作目录、数据集资源和模型配置模板仍需安装准备，归档不包含第三方数据集本体。结果：

| 原始运行 ID | 请求成功 / 总数 | 正确数 | 精度 |
| --- | --- | --- | --- |
| 20261026_152543 | 30/30 | 28 | 93.3333% |
| 20261026_160825 | 30/30 | 29 | 96.6667% |
| 20261026_165049 | 30/30 | 28 | 93.3333% |
| 平均 | — | — | 94.4444% |

[精度确认汇总](evidence/accuracy-confirmed-summary.json) · [原始评测结果](evidence/accuracy-original-results.json.gz) · [环境及轮次元数据](evidence/accuracy-run-metadata.json)。原逐题预测不保留 finish_reason/token usage，不能声称没有输出上限截断；这些轮次也不是三套独立题集。

## Profiling 与性能补充

历史文本为 2048/6144、并发与请求 160；图文为总输入 32768/输出4096、并发与请求36，均成功，16 rank 各10步。[文本验收](evidence/text-profiling-acceptance.json) · [图文验收](evidence/image-profiling-acceptance.json)。

2026-10-29 文本补采：max-model-len=9216、max-num-seqs=48、请求/并发128，128/128成功，输出吞吐2950.13 token/s；96并发轮次未完成，不列通过。采集源数据分别在两机 `/path/to/artifact`。历史原始 trace 不随文档上传。

```bash
# profiler 配置在两机启动时加入；两轮使用独立输出目录
# --profiler-config '{"profiler":"torch","delay_iterations":3072,"max_iterations":10,"torch_profiler_dir":"/path/to/artifact","torch_profiler_with_stack":false,"ignore_frontend":true}'
curl -sS -X POST http://127.0.0.1:31070/start_profile
vllm bench serve --backend vllm --base-url http://127.0.0.1:31070 \
  --model MiniMax-M3.1-Flash-BF16 --tokenizer /path/to/artifact \
  --dataset-name random --random-input-len 2048 --random-output-len 6144 \
  --random-range-ratio 0 --ignore-eos --temperature 0 --top-p 1 \
  --num-prompts 128 --max-concurrency 128
curl -sS -X POST http://127.0.0.1:31070/stop_profile
```

Profiling 使用与上面基线不同的 context/seqs 参数；运行本段前将 serve.sh 对应参数改为9216/48。此段是手工等价入口，完整历史 collector 命令在原工作区采集记录中。硬件效率采用算子耗时加权经验指标，不是墙钟/整卡利用率；AI CPU、通信以及混合通信算子的剔除口径必须随结果注明。

## 已知边界

减层无记录；训练端 replay、不同权重热更新未验证；图文基线确定性与长输入偶发退化尚未定位；完整百万 token 的有效问答未完成。开箱报告中的旧绝对路径和未收录的大型 trace 是历史位置，不能当成此仓库可下载文件。
