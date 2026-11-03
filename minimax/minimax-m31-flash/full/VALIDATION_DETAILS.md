# 全层验证详单：步骤、检查项与原始观测

本页将结论拆分为服务准入、接口功能、状态恢复、权重更新和精度评测。数值表从归档 JSON 提取；未重新上机。启动脚本、全局参数及命令见 [全层复现入口](README.md)，各阶段历史命令见 [完整命令记录](commands/historical-commands.txt)。

## 1. 结果如何判定

| 层级 | 检查项 | 能证明什么 |
| --- | --- | --- |
| 服务准入 | 模型路径、进程、health/models、rank/拓扑 | 服务可用；不能代表回答正确 |
| 功能请求 | 两种文本 API、真实图像、结构合法且非空、未因预算截断 | 指定请求的功能链路可运行 |
| 严格确定性 | 同 DP、同 API、同 prompt 的 text、token、逐 token logprob 逐项精确相同 | 本次请求之间可精确复现；不覆盖所有输入/并发 |
| 状态恢复 | sleep/wake/reload HTTP、is_sleeping、恢复后功能及 HBM | 指定状态恢复链路可运行 |
| 权重更新 | 全量事务结束、权重来源、更新后请求 | 不能用部分 chunk HTTP200 代替完整更新 |
| 数据集精度 | 逐题预测、评测器得分、请求完整性 | 本次指定数据集和采样配置的得分 |

历史报告里整体 strict_determinism_status 与 recovery acceptance 分开：初始图文重复差异不能掩盖，也不能无证据归因于 sleep/reload。

## 2. 参数继承与各特性差异

公共基线：真实60层BF16、双机16卡、TP4 DP4 EP、FULL_DECODE_ONLY、context131072、seqs16、batched8192、memory0.85，未开投机解码及dspark。

| 用例 | 相对基线的配置变化 | 启动入口 |
| --- | --- | --- |
| baseline | 无 | 两机分别 `bash full/commands/serve.sh 0 baseline` / `1 baseline` |
| 路由捕获 | 根据历史命令启用路由捕获附加配置 | 历史 `SERVICE routing_replay`；不能仅启动 baseline 就声称捕获有效 |
| SP | `--additional-config '{"enable_flashcomm1":true}'` | `serve.sh 0 sp` / `1 sp` |
| PP | TP4 PP4 DP1 EP，nnodes2，node-rank0/1，master29502 | `serve.sh 0 pp` / `1 pp` |
| sleep/reload补测 | dev mode、sleep mode、weight_nz_mode0、关闭prefix cache | `serve.sh 0 sleep` / `1 sleep` |
| IPC权重更新 | memory utilization0.8，16推理rank与16 producer rank，packed3072MB | 以历史 `SERVICE npu_ipc...` 和 producer 命令为准；不能套用memory0.85 |

## 3. 基线文本 / 图文

步骤：确认完整checkpoint → 启动双机 → 检查16 worker → health/models → 两种文本API → 真实图像chat。图片是红色正方形和蓝色圆形，测试其形状/颜色描述；不是传一个空URL来验证多模态。

原始 baseline 使用未加模板的 completion prompt 时曾出现重复控制标记。后续独立启动的 NZ0 生命周期实例使用本地 chat template，得到有意义的 completion 回答。不能把后续修正的请求当作原始 baseline 未经记录的成功结果。移植脚本 `rl_request_adapter.py` 显式应用本地 tokenizer 的 chat template；chat API 由服务端处理消息模板。

### 固定逐 DP 请求集合

- 请求头 `X-data-parallel-rank` 固定为0、1、2、3，串行执行，避免无意比较不同副本。
- 每个DP两条文本prompt：`Reply with exactly the word Hello.`、`What is 2 + 3? Answer briefly.`，各经过completions与chat两种API，另有一条真实图文chat。
- 每轮 `4 DP × (2 prompt × 2 API + 1 image) = 20` 条。
- temperature0、top_p1、top_k-1、min_p0、seed0、frequency/presence penalty0、repetition penalty1、max_tokens2048。
- token列表非空，logprob数量与token数对齐且为有限值；文本API记录finish_reason，预算截断不能作为通过。

下表的输出摘要直接来自Level2第一轮恢复后的原始capture，不是新增推理结果。

| DP / 请求 | token数 | 输出开头（截取） |
| --- | ---: | --- |
| dp0:/v1/chat/completions:1 | 15 | 5… |
| dp0:image_chat | 116 | The image shows two shapes on a white background:  - **Left:** A solid red square. - **Right:** A so… |
| dp1:/v1/chat/completions:1 | 15 | 5… |
| dp1:image_chat | 116 | The image shows two shapes on a white background:  - **Left:** A solid red square. - **Right:** A so… |
| dp2:/v1/chat/completions:1 | 15 | 5… |
| dp2:image_chat | 116 | The image shows two shapes on a white background:  - **Left:** A solid red square. - **Right:** A so… |
| dp3:/v1/chat/completions:1 | 15 | 5… |
| dp3:image_chat | 116 | The image shows two shapes on a white background:  - **Left:** A solid red square. - **Right:** A so… |

## 4. 路由捕获与层覆盖

步骤：启用捕获配置 → 发文本双API与真实图文请求 → 解码返回的routed-expert数组 → 核对dtype、token维、层维、Top-K维和专家ID范围 → 对照各层是否有有效路由。

预期数组为 `[tokens, 60, 4]`，不是把每个MoE层压缩成一个整数。范围检查限定0～127，不验证训练程序能消费这些路由，也不代表训练端replay完成。

| 接口 | 路由shape | dtype | 最小 / 最大ID |
| --- | --- | --- | --- |
| /v1/completions | [211, 60, 4] | uint8 | 0 / 127 |
| /v1/completions | [203, 60, 4] | uint8 | 0 / 127 |
| /v1/chat/completions | [211, 60, 4] | uint8 | 0 / 127 |
| /v1/chat/completions | [203, 60, 4] | uint8 | 0 / 127 |
| image:/v1/chat/completions | [375, 60, 4] | uint8 | 0 / 127 |

数组SHA256及当时保存位置见 [路由结果](evidence/routing_replay.json)，层覆盖见 [层覆盖结果](evidence/routing-layer-completeness.json)。数组文件另随本次文档补充归档。数组值合法不等于训练路由语义已验收。

## 5. SP 与双机 PP

**SP**：启用FlashComm1，检查日志确认生效，再运行文本双API与真实图文请求。结果是功能通过；没有对照证明SP吞吐加速或所有并行组合都正确。

**PP**：使用原生mp、nnodes2，而不是把两台机器各启动一套独立PP服务。检查16个 `Worker_PP{stage}_TP{rank}_EP{rank}` 前缀，head运行stage0/1，worker运行stage2/3，各stage四个TP worker。文本双接口和真实图文通过。context配置131072不代表实际发送过131072-token请求，也不代表已测PP数据集精度。

| PP stage | 所在节点 | 期望TP数 | 日志检查 |
| --- | --- | ---: | --- |
| 0 | head25 | 4 | True |
| 1 | head25 | 4 | True |
| 2 | worker26 | 4 | True |
| 3 | worker26 | 4 | True |

## 6. Level 2 与同 checkpoint 重载：完整生命周期

每轮：before capture → `POST /sleep?level=2&mode=abort` → `GET /is_sleeping`确认true → wake weights → reload同checkpoint → wake kv_cache → is_sleeping确认false → health → after capture → repeat capture。共3轮，每轮20条before/after请求。原始steps保存每步时间和状态。

`commands/sleep_determinism_probe.py --action capture|sleep|wake-weights|reload|wake-kv --level 2 --label LABEL --report results/level2.json` 为每一步入口。该probe保存响应，不单独承担所有HBM采集和最终差分；下面是历史验收的结果。

### 逐轮严格比较

| 对照 | 完全一致数 | 总数 | 对齐logprob最大差 |
| --- | ---: | ---: | ---: |
| unchanged-weights-control-0-vs-1 | 16 | 20 | 0.0 |
| unchanged-weights-control-0-vs-2 | 16 | 20 | 0.0 |
| cycle1-before-vs-after | 20 | 20 | 0.0 |
| cycle1-after-vs-repeat | 20 | 20 | 0.0 |
| initial-baseline-vs-cycle1-after | 16 | 20 | 0.0 |
| warm-baseline-vs-cycle1-repeat | 20 | 20 | 0.0 |
| cycle2-before-vs-after | 20 | 20 | 0.0 |
| cycle2-after-vs-repeat | 20 | 20 | 0.0 |
| initial-baseline-vs-cycle2-after | 16 | 20 | 0.0 |
| warm-baseline-vs-cycle2-repeat | 20 | 20 | 0.0 |
| cycle3-before-vs-after | 20 | 20 | 0.0 |
| cycle3-after-vs-repeat | 20 | 20 | 0.0 |
| initial-baseline-vs-cycle3-after | 16 | 20 | 0.0 |
| warm-baseline-vs-cycle3-repeat | 20 | 20 | 0.0 |

初始服务基线与重复请求为16/20；4条差异来自图文。每轮操作前后及恢复后重复则均20/20，三轮before/after累计60/60。Level2恢复通过；服务全状态严格确定性仍失败。checkpoint条目复用同一套恢复证据，不能将其相加当作6套独立循环。

### 显存检查（原始npu-smi MB标签）

| 循环 | 节点 | 设备数 | sleep释放范围 MB | 恢复后相对before增长≤5% |
| --- | --- | ---: | --- | --- |
| 1 | 192.0.2.12 | 8 | 79044～79044 | True |
| 1 | 192.0.2.13 | 8 | 79043～79044 | True |
| 2 | 192.0.2.12 | 8 | 79044～79044 | True |
| 2 | 192.0.2.13 | 8 | 79044～79044 | True |
| 3 | 192.0.2.12 | 8 | 79044～79044 | True |
| 3 | 192.0.2.13 | 8 | 79043～79044 | True |

这些是工具原始标签，不将其直接当作严格MiB；完整before/sleep/after逐卡值保存在压缩JSON。HBM下降证明资源释放现象，不代替权重逐tensor正确性审计。

## 7. Level 1 失败：已做什么、尚缺什么

Level1期望保留CPU权重备份，wake恢复该备份，不重新从checkpoint加载。历史首次sleep与最新复测均失败在 `camem.py:216 torch.empty(...,device="cpu",pin_memory=True)`；`aclrtMallocHostWithCfg` 返回207001，HTTP500，未完成wake，因此不能提供恢复后通过结果。

复测检查：失败worker memlock上限64MiB，失败后两机仍有约1.76/1.55TiB普通可用内存；停止服务后25机单进程申请1、4、8GiB pinned memory全部成功。由此不能仅凭64MiB上限认定原因。缺失证据为失败单次申请大小、各worker累计备份量、全机驱动锁页配额与并发分配过程。

复测首次启动还在sleep调用前出现HCCL watchdog / MoE dispatch561000；第二次启动通过预热后才复现207001。两类故障发生阶段不同，不能合并为同一根因。

## 8. IPC 更新失败的事务边界

原用例为真实完整checkpoint的原生NPU IPC传输，16推理worker及16个producer rank；packed buffer3072MB，memory utilization0.8。握手、checkpoint manifest一致性、sleep2、wake weights已通过；10次部分stream请求返回200，但全量传输没有完成。

接收端 `packed_tensor.py:321` 执行 `raw.contiguous().view(dtype).view(*shape).clone()` 时OOM：该次请求36MiB，日志报告剩余26.96MiB。日志allocated/reserved统计存在大于reported total的值，保留原值，不能从中推导可靠的显存碎片比例。clone用于保护producer复用buffer，不能直接删除并宣称修复。

更新后两API检查为未执行；`real_checkpoint_outputs_identical`是原验收目标字段，**不是已达成结果**。后续应记录实际producer/receiver显存、chunk大小、缓冲区生命周期，再补完整权重更新及输出对照。

## 9. HCCL、DCP、PCP 的阻塞判定

HCCL用例要求独立物理producer NPU，推理已占满16卡；没有单独生产者资源，未执行完整权重事务。不能用IPC同卡生产者替代该前置条件。

DCP/PCP的当前MiniMax sparse backend明确不支持，不以修改开关强行启动当作通过。这里是实现范围受限，与实际已执行但失败的Level1/IPC不同。

## 10. 全层数据集精度：复现细节与完整性

AIME2026使用30道题、0-shot chat prompt、temperature0、seed1234、batch128；三轮使用相同题集，max_out_len122880，服务context131072。平均值根据整数正确数计算为(28+29+28)/(30×3)=94.4444%，样本标准差为1.9245个百分点。不是将已经四舍五入的百分数再次平均。

配置脚本在修改前检查path、host_ip、host_port、batch_size、max_out_len、generation_kwargs各项只匹配一次；若模板版本不一致则失败退出，避免静默使用旧配置。`ais_bench --models vllm_api_stream_chat --datasets aime2026_gen_0_shot_chat_prompt`每轮新建输出目录。

三轮请求分别30/30成功、失败0，correct分别28、29、28。原始预测未保留finish_reason与token usage，无法确认是否存在输出预算截断。没有官方同设置BF16基线或置信区间验收阈值，不将94.44%自动称为客户已接受。

## 11. Profiling / 长序列为补充测试，不代替RL验收

文本2048/6144，context9216、总请求和并发160；图文总输入32768/输出4096，context37888、总请求和并发36，图片256×256。两项16rank中段10步，未出现KV抢占。四DP负载并非完全均匀，不能将总并发直接当作每DP相同batch。

2026-10-29补采文本128/128、seqs48：输入总262144、输出总786432，128/128成功，benchmark266.58秒，输出吞吐2950.13 token/s，16rank完成解析。96并发未完成。硬件效率1.0781或保留MoE dispatch/combine后0.9628是特定算子加权经验指标，非整机利用率。

长序列1,000,000 token ID仅验证容量/计数；中文560793与976530 token两接口有语义成功结果，近1M同输入4次有1次退化重复，原因未定。不能据此宣称所有百万token功能通过。

## 12. 下一步验收门槛

| 项目 | 缺失项 | 补验通过应具备的证据 |
| --- | --- | --- |
| Level1 | driver pinned allocation根因及wake | 分配规模日志、连续sleep/wake、逐DP输出、HBM |
| IPC | 完整传输和更新后验证 | 所有chunk结束、权重来源、双API输出和buffer生命周期 |
| HCCL | 独立producer卡 | 实际拓扑、完整事务、更新后可用性与输出 |
| 不同权重更新 | checkpoint A/B切换 | A→B→A、预期可观测变化及恢复；不能只同权重reload |
| 训练replay | 捕获数组被训练消费 | 生产/消费协议及训练端检查 |
| 图文一致性 | 首次/后续差异原因 | 固定DP及相同输入的独立对照，不混入RL归因 |
| 减层 | 可核验减层资产及运行记录 | 资产哈希、层号、拓扑、命令、逐项请求和结果 |

原始JSON中的运行路径是历史环境位置。报告的精确数值来自已归档证据；本次新增文档和相对路径调整没有经过新一轮NPU验证。

## 13. 逐接口原始记录补充

- [baseline-text.json](evidence/baseline-text.json.gz)
- [baseline-image.json](evidence/baseline-image.json.gz)
- [sp-text.json](evidence/sp-text.json.gz)
- [sp-image.json](evidence/sp-image.json.gz)
- [pp-text.json](evidence/pp-text.json.gz)
- [pp-image.json](evidence/pp-image.json.gz)
- [routing_replay-text.json](evidence/routing_replay-text.json.gz)
- [routing_replay-image.json](evidence/routing_replay-image.json.gz)

Level1最新复测：[25机错误](evidence/sleep1-retest-192.0.2.12-sleep-errors.json)、[26机错误](evidence/sleep1-retest-192.0.2.13-sleep-errors.json)、[1/4/8GiB分配对照](evidence/sleep1-retest-pin-probe-25.json)。
