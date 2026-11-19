# Ascend 适配最新 Direct：实施建议

多人实施分工、接口依赖与集成门槛见 方案三优化后的并行执行计划（历史材料未附，原引用 `方案三-多人并行适配执行计划-20261117.md`）。本文继续说明各技术域的具体适配内容。

补充参考：[RFC54477 权重热更新重构与 Ascend 适配](RFC54477-权重热更新重构与Ascend适配.md)。其中ND staging、strict refill及私有函数patch属于候选设计，当前基线的存在性与语义差异见并行计划第1.1节；不将其历史状态/实验声明当作当前已验事实。

基线：vLLM `3d836e13ea80247f0f7a2a4133a63e32d379fa0f`；vllm-ascend `b53996aad5d959f0b4511b4b87aadb2eb425b045`。本文是源码对照与建议，不是代码适配交付或 NPU 测试结论。两仓没有本轮新增代码修改。

当前优先级（2026-11-17，按用户最新要求）：**P0 为 BF16 × eager/图模式 × IPC/HCCL × packed/unpacked；P1 为草稿模型与投机推理；P2 为量化（预量化 checkpoint 与 online quant）。** 此分级替代此前 P0–P3 的安排，也替代旧方案中“draft/MTP 不纳入当前阶段”的总范围限制。P0 的首个增量可以 target-only，但总体计划必须包含 P1。

## 1. 结论与边界

复用上游 direct core、layout 记录、参数工具及统一 PWAL/hook 协议。在 Ascend 自有 worker、通信数据面、量化/attention、allocator 和图管理处实现等价契约。不要复制一份 direct.py 或 fork 一套通用加载状态机，也不要把“能进入 begin”作为已支持。

先以 target-only、MRv2、BF16 最小模型打通链路，再在 P0 内完成 eager/图模式、IPC/HCCL、packed/unpacked 的组合验证；随后实施草稿模型/投机推理，最后做量化。BF16 可达路径所需的 NZ、融合/转置和派生状态保址属于 P0 基础正确性，不随量化专项后移。首批可用组合之外，必须在修改模型前拒绝不受支持的 direct 配置，保留已有 layerwise 行为。

P0 至少覆盖以下 8 种组合，不把其中一种真实通道的结果代替其他组合：

| 执行模式 | IPC unpacked | IPC packed | HCCL unpacked | HCCL packed |
| --- | --- | --- | --- | --- |
| BF16 eager | 必验 | 必验 | 必验 | 必验 |
| BF16 图模式 | 必验 | 必验 | 必验 | 必验 |

图模式还需登记实际采用 FULL、PIECEWISE 或其他受支持模式；通过一个模式不代表全部图模式已验。eager 验证数值、会话和资源；图模式额外验证原图引用、地址与 capture 次数。代表模型/硬件/并行配置固定到支持表，8 种组合不意味着无边界覆盖所有模型与硬件。

## 2. 可直接复用的已有资产

| 资产 | 本地状态 | 处理 |
| --- | --- | --- |
| `reload/direct.py` / `lifecycle.py` | 最新上游已合入 | 沿用公共实现，NPU 只补所需扩展点 |
| HCCL / NPU IPC 引擎与 trainer factory 注册 | 已有 `hccl`、`npu_ipc` backend | 复用控制面和注册，无需新增 direct 专用 backend |
| HCCL start/finish | 从 NCCL engine 继承 | 已能分派到 direct；仍需补 worker 上下文和数据面时序 |
| MRv2 load_model | 调用父类加载流程 | 核验 PWAL 前已记录 layout，不再重复 record/finalize |
| `replace_parameter`、`refill_derived` 等 | 上游已有，部分 NPU MLA/MXFP8 已使用保址实现 | 复用后补缺口，不重做已实现分支 |

建议继续使用 `VLLM_RELOAD_MODE=direct`，配套 `backend=hccl` 或 `npu_ipc`；不要增加另一个开关表达同一 reload mode。

## 3. PR-A：worker/runner 接口与会话，最高优先级

入口：[NPUWorker](https://github.com/vllm-project/vllm-ascend/blob/b53996aad5d959f0b4511b4b87aadb2eb425b045/vllm_ascend/worker/worker.py)，约 398–438 行。

已定位的静态缺口：

1. `finish_weight_update()` 调用 `self.model_runner.reset_lora_state()`。当前两仓全文搜索没有该方法定义/赋值；MRv2 继承的上游 runner 提供的是 `on_target_weights_replaced()` 等接口。预期可能在 engine finish 已成功后触发 AttributeError，需要真实 worker 用例确认。不能用 `hasattr` 静默吞掉通知，而应明确对应的缓存/派生刷新职责。
2. engine finish 抛异常时，尾部 `_weight_update_active=False` 不执行，下一次 start 被已有会话挡住。
3. Ascend 不覆写 `has_incomplete_weight_update()`；[WorkerBase](https://github.com/vllm-project/vllm/blob/3d836e13ea80247f0f7a2a4133a63e32d379fa0f/vllm/v1/worker/worker_base.py) 的默认实现始终返回 False。更新尚未结束或失败时，对外状态不符合当前上游语义。
4. start/update/finish 没有一致包住 `set_current_vllm_config` 和 NPU device 上下文。HCCL update 自己绑定 device，但其他路径不能因此免查。

实施：新增与支持范围匹配的未完成状态，begin 前保守标记、成功 finish 后清除；抽出会话结束 helper，负责 active/target/实际借用资源的清理；direct finish 失败结束旧会话并保留未完成状态，允许新完整更新修复；保留 layerwise 原语义。对齐目标 runner 的通知接口及 cache 管理，不照搬 GPU 的 CUDA-only KV loan 实现。

P0 的 target-only 增量可以暂时明确拒绝 draft 更新，但 API/状态设计要兼容 P1 的 target/draft 目标隔离，不能丢失 target reset 和失败修复语义。磁盘 `reload_weights` 与 transfer 路径分别核对，不用一个入口测试代替另一个。

验收：真实 worker start/update/finish；finish 中 PWAL 异常；新完整更新；未完成状态上报；非法时序；成功通知次数；layerwise 对照。恢复后仍需对照冷 B 与原图，状态可重开不等于数值已修复。

## 4. PR-B：NPU 参数、派生状态与物理 format

对所有首批可达 PWAL，按 checkpoint 参数、运行权重、派生值、guard、物理 format、消费者逐行登记。

本组拆为 B0（P0，非量化权重、BF16 派生与必需 NZ）和 B1（P2，预量化 scheme）。下表中的 PDMix、LAOS、W4A16-MXFP4、MXFP8、KV C8/MXFP、FlatQuant 保留审查记录，实施与专项验收后移，不作为 BF16 首期发布的默认前置条件；必须确认它们不会被首期配置选中。

| 源码 | 实际看到的风险 | 修改方向 |
| --- | --- | --- |
| `ops/linear.py:107` | NZ 转换、3D reshape/contiguous 后 `.data` 重绑；FP32 派生已有部分 replace_parameter | 保留加载身份；每轮 runtime 地址稳定；逐分支验证，避免重复转换 |
| `ops/fused_moe/routed_experts.py:105` | 强制 NZ、expert clone/list、某分支 `del layer.w13_weight/w2_weight` | 保留 checkpoint 入口；持久 list/元素回填；删除换成与内核读取匹配的注册/retire 策略 |
| `w8a8_pdmix.py:90` | 普通属性 `weight_scale_fp32` 每次重建 | 原位刷新该属性及其实际消费者 |
| `w4a4_laos_dynamic.py:88`、`wna16/w4a16_mxfp4.py:142` | INT4 pack、解包、NZ 与转置重绑 | 设计持久运行缓冲，验证容量/offset/format/pack 顺序 |
| `w8a8_mxfp8.py` | 父类已有 buffer/list 保址；DS 子类独立覆写 | 父类复用，子类独立核对 E8M0 展开与转换，不能继承父类 PASS |
| KV C8/MXFP C8、FlatQuant | reciprocal、offset、E8M0 原始编码、host scalar | 派生状态回填；验证编码；图固化的 host scalar 变化需可更新机制或明确拒绝 |

guard 的清理要放在每轮 PWAL 前，而非每个 bucket；模块 guard 和 scheme 自身 guard 都需核对。普通 `.data=` 不能一律判错，view 和真正换 storage 需分别处理。

**NZ 决策：** 不把通用 `storage_view` 对逻辑 shape 的解释当作内部格式正确性证明。优先探测原存储同 format 回填；不成立时，可研究“保留 checkpoint ND 参数 + 持久 NZ runtime tensor”的 NPU 局部策略。后者增加常驻内存并改变 apply 消费入口，不能宣称零额外权重存储；需实际 NPU copy/format/算子验证后决定。`quantization/nz.py` 的 NzCastSpec 可以复用来保持 C0 recipe 一致。

`weight_nz_mode=0` 只是首批配置限制，不能证明实际 ND；FUSED_MC2 存在强制 NZ 分支。其真机验收仍属于原 G1 必须完成的工作。

## 5. PR-C：统一 hooks 与最新 indexer/MLA 加载，必须显式适配

最新上游 [init_indexer_wk_weights_proj](https://github.com/vllm-project/vllm/blob/3d836e13ea80247f0f7a2a4133a63e32d379fa0f/vllm/model_executor/layers/sparse_attn_indexer.py)（约 1743 行）已将：

```text
checkpoint 可见：wk、weights_proj
运行态融合值：wk_weights_proj.weight（普通属性，不是 checkpoint Parameter）
finish：fuse_indexer_wk_weights_proj 原位融合
```

但当前 Ascend [IndexerWrapper](https://github.com/vllm-project/vllm-ascend/blob/b53996aad5d959f0b4511b4b87aadb2eb425b045/vllm_ascend/ops/mla.py)（约 41 行）只显式接管 `wq_b`、`wk_weights_proj`、`k_norm` 和原模块的直接 Parameter，没有接管新 `wk/weights_proj` 子模块，也没有继承原 indexer 的融合 hook。因此使用新上游 indexer 的模型路径可能丢失 checkpoint 路由或融合刷新；具体模型可达性仍需运行验证。

实施方向：wrapper 暴露新 checkpoint 子模块并保留共享关系；将融合执行权明确放在正确的无参 hook，后端继续消费融合运行值；不将 fused 普通属性重新注册成 checkpoint Parameter。不能简单把整个旧 indexer 挂回 wrapper，导致路径前缀变化或重复注册。

Ascend 自有 GLM5-next 仍使用 `wk_weights_proj` 参数和 `_pending_indexer_wk_fp8` 配对 helper（`models/glm5next/model.py` 约 837、1160 行）。需迁移成独立加载→PWAL 反量化/融合，或者提供等价的 NPU 层契约。仅因旧 helper 能跨 bucket 暂存，不能认定符合新生命周期：未完成配对、接收 buffer 复用和跨轮清理都需处理。Kimi 等仍使用自有融合路由的模型逐个对应，不能全局删除旧映射。

优先级上，PR-C 先完成新版子模块接口、BF16 的加载/融合和 hook 所有权；FP8 weight/scale、反量化及量化 KV 等分支归 P2。若某模型只能通过这些量化分支运行，将该模型支持后移，不能删掉它依赖的转换后宣称 BF16 已支持。

**还需解决 hook 双调：** `AscendSFAIndexerBackend` 与 `Glm5NextKPoolIndexerBackend` 都是 nn.Module；IndexerWrapper 也显式委托 `impl.process_weights_after_loading()`。公共 children-first 遍历可能先调 impl，再调 wrapper，造成重复；如果父 attention 带 quant_method，子树又可能被公共遍历跳过。应指定每个 hook 的唯一所有者，分别验证无 quant_method 和 KV quant method 两种树形，不能仅改签名。

KPool 的 `_key_weight/_gate_weight/_head_weight_f32/_norm_*` 在约 858 行重新赋值，forward 中还存在 fallback。改为冷启动/finish 完整回填持久值，并证明图消费的是该值；不要依赖 forward 临时修复。

## 6. PR-D：HCCL/IPC 数据面及可选能力约束

保持已有继承，不另实现 direct engine。核对并补齐：

- start 的 padding 清零流→packed 接收/加载流的先后关系：接收流等待 caller 事件/流，finish 等待全部写入完成。现有 broadcast 后同步不能代替 begin→load 的顺序。
- packed 混合 dtype/非对齐 entry、buffer 复用、异常 drain 和存活；已有 byte clone-unpack 可复用。
- NPU IPC 的设备绑定、逻辑卡号/UUID、句柄和导入缓冲生命周期；与 worker device context 一起验证。
- HCCL 重 init 的 metadata/plan 状态；模型目标切换和失败通道重建。
- **load plan 当前不能直接声明支持：** HCCL 覆写 init/receive，没有接上父类记录 plan metadata、manifest 检查及 `load_planned` 的完整路径。首期在 worker/trainer init 提前拒绝 `use_load_plan=True`，后续再专项接入，不能静默忽略。
- 当前 HCCL trainer 明确只支持 group rank 0 发送，`_broadcast` 拒绝 `src!=0`，receive 则固定 src 0。将这一限制提前验证并写入支持表，不声称继承了 NCCL 多源广播能力。

验收先真实非 packed TP1，再 packed、TP2；IPC 单列。不能把磁盘 bucket 或 mock broadcast 通过记为真实 HCCL/IPC 通过。

## 7. PR-E：sleep2/CaMem 与图生命周期

这是保留的配套工作，不因曾列在 P1 就挤占用户指定的第二优先级。P0 默认先不启用 sleep2；若某项 P0 支持声明包含 sleep2，则本节相关正确性要求必须随该声明一起满足。具体 sleep2 交付位置单列，不作为完成全部 P0 八组合的默认前置依赖。

当前 worker 主要备份 named buffers/显式保留参数；CaMem 按 allocation tag 备份/丢弃，未实现上游“只丢弃可重写纯参数 allocation”的完整等价契约。

实施：从实际 allocation/active blocks 快照判断纯参数、混合块与未知块；按 storage 处理别名与显式 retain；混合/未知块采用保守保留；与 worker 的旧备份职责去重。明确 weights/runtime/graph/KV tag 的映射、备份及唤醒条件，pool 保活与释放。不能只把普通属性改为 buffer 就认为所有共享块/指针表都已覆盖。

`device_allocator/sleep_mem_optimized.py:122` 会清图，wake 时 capture_model。首期同图配置必须在 sleep 前拒绝该冲突模式；支持它需要另外设计可保留的图/workspace/HCCL 资源路线。保留现有 cleanup+recapture 作为独立行为可以，但不能算 direct 同图验收通过。

联合验收：sleep2→wake weights→start/update/finish→wake KV；A→B→A 与冷 B；图不重捕获；普通非参数状态、padding、混合块、CPU/NPU 内存稳态与通信恢复。

## 8. PR-F：online quant，独立扩展

优先级 P2，与其他量化适配同属第三优先级；组内先后按目标 scheme 的依赖决定，不作为 P0/P1 前置依赖。

最新上游 `online/loading.py::checkpoint_method` 硬编码选择 FP8/ModelOpt/compressed-tensors 等 checkpoint methods 和量化函数；Ascend 当前按自己的 quant config/registry 选择 scheme。这两套选择不能仅靠 import 或打开环境变量视为接通。

先确定需求是 BF16 trainer→BF16 inference，还是 BF16→ModelSlim INT8/FP8/MXFP 推理。后一种需要：NPU online scheme→checkpoint method 映射、NPU 量化算子、values/scales 的原 loader 路由、merged shards/TP block 对齐与 scale 归约、跨 bucket配对及 buffer 存活、同一冷/热加载语义。避免运行上游 CUDA-only quantizer。

建议争取上游提供 checkpoint-method/quantizer 的平台扩展点；在扩展点可用前，Ascend 局部实现并记录撤销条件，避免复制整套 OnlineQuantizationConfig。若方案不能映射，初始化时明确拒绝。先只做一个目标 scheme，再扩 MoE/NVFP4；load plan + online 仍遵守当前上游拒绝规则。

## 9. PR-G：草稿模型与投机推理，第二优先级 P1

先建立 BF16 的 draft/投机链路，不将草稿 lm_head 量化等 P2 能力绑定为 P1 必需条件。按选定支持的独立 draft、MTP、DFlash/DSpark 等实际方法逐项登记，不由一个方法通过推出全部通过。

实施工作：

1. NPUWorker 增加 `supports_draft_weight_updates`、`start_draft_weight_update`、draft model 获取与 engine target 切换的等价实现；每轮成功/失败都恢复默认 target，分别跟踪 target/draft 未完成状态。
2. 正确区分 target 更新、draft 更新和需要两者配套更新的流程；target 更新后刷新由 target 派生的 draft 状态，不误写冻结的独立 draft。
3. 处理 embedding/lm_head 的共享与别名，复用上游 owner/共享布局约束；不通过重复 PWAL 改坏另一模型的 storage。
4. 接入 Ascend speculator 与 draft 图管理器，刷新 fused K/V、norm、映射和其他 weight-derived 缓存；区分持久权重派生与旧请求 KV/投机状态，后者按更新边界重置或隔离。
5. 分桶更新的完整性判断放在正确的事务结束边界，检查目标模型方法是否在单 bucket 内提前校验/收尾；仅继承已合入能力，不把仍开放的上游修复当作已可用。

验收：target-only 更新、draft-only 更新、配套更新；共享/独立权重；eager/图模式；复用 P0 的 IPC/HCCL 与 packed/unpacked 设施，新增组合逐项记录；draft 更新失败不得误改 target，新完整更新可修复。数值与匹配配置的独立冷加载对比，记录投机接受率、接受长度及性能，不能以接受率单一指标替代正确性。

## 10. 合并/验收顺序

| 阶段 | 工作组合 | 可以声称什么 |
| --- | --- | --- |
| S0 / P0 起步 | PR-A + PR-B0 最小 BF16 + 基础 refit | 先打通 eager 与图下的真实更新；单组合通过只代表该增量 |
| S1 / P0 完整交付 | PR-B0 + PR-C 的 BF16 必要分支 + PR-D | BF16 × eager/图 × IPC/HCCL × packed/unpacked 八组合；指定模型/硬件/并行的明确支持范围 |
| S2 / P1 | PR-G + worker/runner/hooks 的草稿增量 | 选定草稿模型/MTP/投机方法的权重更新、目标隔离、派生刷新与图正确性 |
| S3 / P2 | PR-B1 + PR-C 的量化分支 + PR-F | 预量化 checkpoint 与 online quant 按目标 scheme 逐项扩展，不影响 P0/P1 已支持组合 |
| 配套项 | PR-E；load-plan 等性能项另外评估 | sleep2/CaMem、额外资源释放与性能优化单列；承诺相应组合时必须完成联合验收 |

对照原方案：PR-A/B 属 G1，PR-C 属 G2，PR-D/E 属 G3；PR-G 恢复为明确的草稿/投机专项，不能再沿用旧方案“排除 draft/MTP”的总范围说明；online 仍单列扩展组，但优先级统一归 P2。refit/失败验证从 S0 横贯所有阶段，每个 PR 保留唯一来源归属，联合发布用累计栈。

统一验收：变化权重 A→B→A、独立冷 B、具名 kernel 消费者的地址/alias/layout/物理 format、图捕获次数、真实入口与真实通道、失败修复、50–100 轮稳态资源趋势（建议，按模型成本调整）。不以“参数指针没变”“与 layerwise 输出相同”单独替代 oracle。

建议首个代码迭代优先做 PR-A 和 BF16 的 eager/图双模式 refit，随后完成两种通道与两种打包方式，过程中补齐所选 BF16 模型的 indexer/派生契约；之后再做 PR-G 与量化。量化分支仅做必要的可达性/拒绝边界盘点，不抢占 P0/P1 实施主线。

本次按最新要求调整范围和阶段依赖。此前 6–8 周估算对应旧组合，不能直接作为新增 P1 草稿/投机范围后的总周期；后续按 P0 八组合、选定 P1 方法和 P2 scheme 分别估算。非量化必需 NZ、真实通信和图生命周期仍可能影响 P0 关键路径。
