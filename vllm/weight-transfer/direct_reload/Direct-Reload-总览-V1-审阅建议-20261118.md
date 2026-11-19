# Direct Reload 总览 V1：审阅建议

审阅对象：[Direct-Reload-总览-背景与适配计划-V1.md](Direct-Reload-总览-背景与适配计划-V1.md)。日期：2026-11-18。源码基线：vLLM `3d836e13ea80247f0f7a2a4133a63e32d379fa0f`，Ascend `b53996aad5d959f0b4511b4b87aadb2eb425b045`。本次未改 V1 原文、未更新仓库、未运行模型；行号对应所读的 447 行版本。

总体结构可以保留：背景→链路→适配清单→优先级→并行计划。主要需要修正的是将“正确性目标/候选方案”写成“当前实现保证”，以及波次依赖和 P0 验收范围中的矛盾。

## 1. 必须纠正的技术表述

### T1. “放不下就报错”不是 replace_parameter 的当前行为

位置：V1 第 124、178、225 行。

源码：`model_executor/utils.py:126::_in_own_storage` 在容量、device、对齐等不满足时可直接返回 `new`，上层随后可重绑 `.data`，不是一律 raise。当前公共工具没有通用的“图引用地址改变就拒绝”保证。

建议统一替换为：

> `replace_parameter` 在布局匹配或旧 storage 可容纳结果时尽量原位写回；不满足条件时可能重绑 `.data`。图模式适配必须额外保证每个 kernel-visible 对象在更新后保持捕获地址；无法保址的配置需拒绝或重建，不能将公共工具的行为当作完整保址校验。

### T2. storage_view 本身不检查容量

位置：第 231 行；第 124 行可配套澄清。

`utils.py:113::storage_view` 明确说明 `set_` 不做该处所需的 bounds check，由调用方检查 view 是否装得下。`direct.py::_checkpoint_views` 做了容量检查，不能把责任归到 helper。

表格建议写：

> `storage_view`：在指定 storage 上建立 view，不负责容量验证；调用方（如 `_checkpoint_views`）先校验跨度、offset 与容量。

### T3. 两个候选 helper 被误列为上游现有工具

位置：第 229–230、290 行。

当前两仓未找到 `refill_expert_list`、`refill_derived_buffer` 定义；第 290 行尤其将前者列入 `model_executor/utils.py` 的“天然可用”清单。应从上游已有工具中移除，另列“拟新增 Ascend 工具/待探针”。专家列表需要处理不等于此名字的通用工具已经存在；已有 MXFP8 类的局部列表更新方法也不能当成同名公共 API。

### T4. hook 遍历不存在文中所画的独立 deferred-attention pass

位置：第 215–219 行。

当前代码是 quant-method pass，随后进入 `weight_utils.py:1492::finalize_model_weights` 的 children-first。带 quant_method 模块的子树不递归；QuantizeMethodBase 实例跳过；deferred attention 的判断用于选择 `device_loading_context`，不是先单独执行再在遍历中跳过。

建议替换为：

| 顺序 | 行为 |
| --- | --- |
| 1 | 公共 PWAL 入口遍历 quant methods，调用各方法的后处理 |
| 2 | `finalize_model_weights` children-first 调用 module 无参 hook；共享模块去重；quantized 子树由其所有者负责覆盖 |
| 调用上下文 | deferred attention hook 在需要时进入 device loading context，不是独立的一轮 pass |

“必须删除所有 wrapper 转发”也应改为“只有公共遍历已覆盖的注册子模块才删除重复委托；未被遍历的 backend/quantized 子树仍需明确委托”。backend impl 的带 dtype 方法不必统一改无参。

### T5. 当前时序会让读者重复 start

位置：第 128、187–191 行。

V1 先讲 worker 已 start、begin 返回，然后才让 trainer 调 `engine.send_weights()`。但 `send_weights()` 自己先调用 `client.start_weight_update()`，照此操作会重入 start。

建议把 2.3→2.5 的外层起点统一放在 trainer `send_weights`：

```text
caller pause/drain
  → trainer engine.send_weights()
      → client.start_weight_update()
          → worker/engine → start_reload → begin
          → 若开启 load plan，在这次 start 内完成首次 plan 构建
          → lifecycle/worker 各自维护相应会话标记与资源
      → update RPC 与 broadcast 并发
          → 接收 → 原 loader 写入
      → client.finish_weight_update()
          → finish_reload → PWAL/hooks → 会话结束
  → 仅成功后 resume
```

细节：`NCCLWeightTransferEngine.start_weight_update` 在 `start_reload` 后、返回 worker 前构建 plan；worker 的 KV 资源借用与 active 标记发生在 engine start 返回后，不能把这些步骤随意换序。

`send_weights` 建议称“仓库推荐的 trainer 侧编排入口”，而非“唯一发送入口”；仓库也存在不导入 vLLM 的 trainer 协议示例。

### T6. U8 过度声明 HCCL/IPC 自动继承 receive/packed/plan

位置：第 295 行，与 E1/J1 的“未接通”相矛盾。

Ascend 覆写了 HCCL 的 init/receive、IPC receive，并使用自己的 packed helper，因此不会自动执行父类这些方法内的逻辑。可复用的是未被覆写的生命周期、公共 wrapper 和上层控制协议；数据面与 plan/metadata 要逐项对照。

建议写：

> HCCL/NPU IPC 已有 subclass 接入，可复用未覆写的生命周期和公共控制接口；本地 init/receive、packed、设备/流和 metadata/plan 分支需独立适配与验收，不因继承关系自动获得。

### T7. “incomplete 状态用于崩溃重启后识别”不准确

位置：第 307 行。

这是当前 worker 进程内的未完成更新状态，不是持久化恢复记录。应改为：

> 使当前进程能上报“更新已开始但尚未成功结束”，避免失败后被误判为就绪；进程崩溃/重启需另行重建与恢复策略，不能依赖该内存标记。

### T8. packed 的分配和 clone 描述混淆上游与 Ascend

位置：第 128、158 行。

当前上游 NCCL unpack 默认返回 packed storage 的 views；consumer 每次 receive 分配 packed tensor，并通过 stream/event/record_stream 保持顺序和存活。Ascend 本地 HCCL helper 有 byte clone-unpack，IPC 又有自己的复用处理。不能在上游链路概述里统一写成“byte clone-unpack”“固定几个底层 storage 轮转”。

建议写：

> packed 按配置数量轮转并发 slot/stream，并将多个 tensor 合成 chunk 传输；底层 buffer 是否复用、是否 clone，依 backend 的实际实现而定。上游和 Ascend 的两条数据面分别核对。

### T9. FAILED 需区分拒绝阶段与损坏阶段

位置：第 66–67 行。

“任何环节失败均半更新”过强。begin 的前置拒绝可能尚未修改模型；load/PWAL 失败才可能留下部分覆盖/混合布局。worker 仍可保守维持未完成状态，状态策略与字节是否已变化应分开。

建议写：

> 前置检查拒绝时，按实际状态决定是否需要修复；begin 改写后、load 或 finish 失败时保持不服务。可恢复的数据/后处理故障通过新完整更新修复；communicator、stream 或 rank 故障可能需要重建引擎/进程。

### T10. 主链路需限定为普通 checkpoint 路径

位置：第 59、130、206–208 行。

“只有三段API”“没有 direct 专用 loader”“每个模块全量发送”可作为简化，但要限定层次：磁盘 `reload_weights` 不必调用传输三段 API；load plan 可绕开部分模型 loader；online quant 在建层时包装 weight loader；draft 有独立 start。发送契约是完整覆盖每个必需的非幂等/变布局模块及 padding 参数，不是无条件发送所有冻结/无变换张量。

建议 2 章开头补一句：

> 本章先描述普通 checkpoint-format、target 模型、未启用 load plan/online quant 的上游路径；磁盘、plan、online 和 draft 的差异另列，不把逻辑阶段与传输API混为一谈。

## 2. 计划与验收建议

### 补充：第 2 章另外三处边界

- 第 158 行“后续轮自动回退 load_weights 修复”应改为“plan 失败清除 metadata 并抛错；调用方保持暂停，显式发起新的完整更新，成功后才能恢复服务”。代码没有自动补发数据或自动恢复，重新规划还需 re-init。
- 第 118 行参数身份检查主要检测同名加载 Parameter 被替换，未通过原始名字集合差检测所有删除/改名；不能写成任意删除都由这一个检查拒绝。
- 第 128、134、162 行统一区分 API update 调用、source slice、wire chunk。一次 update RPC 内可以消费多个 packed chunk，不能与一个 bucket 等同。

### P1. W1 首个验收点不能只依赖 A1/A2

位置：第 353、379–382 行。

修好 worker API 不能单独证明完整更新正确。S0 至少依赖 A 的最小会话修复、选定最小 BF16 模型的正确加载/后处理、E1 真实 unpacked 通道，以及 V 的冷 B oracle。B 的全部复杂模型不必提前完成，但“最小模型这部分 B”不能省略。

建议写：

> A1/A2 是当前真实 worker 成功闭环的阻塞之一；B/E/V 可以并行做局部验证。S0 在最小 A+B+E+V 累计栈就绪后验收。

### P2. 支持限制入口应随 W1 一起交付

位置：第 389 行。

W1 已有可用增量，A4 若直到 W2 才生效，之前不受支持配置可能进入改写模型阶段。改为“随首个可用增量启用支持 allowlist/拒绝；W2 随覆盖扩展更新支持表”。

### P3. W0 不应变成整周等待，D1 不是唯一不确定项

位置：第 355、372–375、404 行。

把“首周，无功能代码”和“唯一能改变后续计划”写死，会人为串行化。IPC no-sleep 的内存预算、图模式实际可达性、HCCL 跨流、模型 hook 等都可能改变范围/排期。且 CaMem 是配套工作，不应占用 P0 唯一关键人力。

建议写：

> 开工阶段先冻结最小契约、基线与 oracle；NZ 是主要技术风险之一，与 A 的确定性接口修复、E 的最小通道和 V 的测试改造同步推进。CaMem 探针在配套资源允许时进行，不作为 P0 开工门槛。W 表示依赖波次，不代表固定周数。

第 386 行将“按 W1 探针结论”统一为“按 W0/开工期探针结论”，避免前后日期矛盾。

### P4. E1/E2 并行需要明确两名执行者

位置：第 360–365、406 行。

“E线”若只有一人，HCCL packed 与 IPC 只是可并行任务，不会自然并发。建议标出建议人数：A1人、B1人、E-HCCL1人、E-IPC1人、V1人，共5人；4人时两通道交错推进。公共 packed 文件仍指定一个统筹 owner，禁止同 hunk 双写。

“B和E完全互不依赖”改为“契约冻结后可并行开发、局部验证，最终共享 worker/stream/模型状态的集成契约”。

### P5. sleep2 不能放在 P0 所有组合的最低要求中

位置：第 334、428 行。

建议将资源验收拆为：

- P0 必验：真实通信、跨流、声明支持的 TP、NPU/CPU 峰值和稳态内存、更新时延。
- sleep2 附加：tag/wake、纯参数/混合 allocation、普通非参数状态、图生命周期；仅在声明支持 sleep2 时必验。

避免 3.3 写“P0默认不启用”，4.5 又把它作为所有证据的最低要求。

### P6. 双区 NZ 内存不是固定“2×”

位置：第 406 行。

建议写：

> 若采用 ND staging + NZ runtime，相关权重的常驻内存为 ND加载区字节数 + NZ运行区实际物理字节数，另有clone/cast临时峰值；受padding、dtype、C0、TP/EP和处理并发影响。需按目标配置实测，不固定为整模型或每层恰好2×。

### P7. 八组合之外补一个“支持范围表”

位置：第 331、338–343、430 行。

建议每条验收结果绑定：两仓 SHA、模型/checkpoint及A/B标识、硬件/CANN/torch_npu、dtype与实际quant method、TP/EP、execution mode、图模式/capture sizes、backend、packed、sleep、load-plan、运行tree、日志。无须一开始展开全部笛卡尔积，但不能用一个代表模型八格通过宣称所有BF16模型通过。

图模式还应证明实际 replay、没有 eager fallback，旧 storage 保活以防释放后同址重分配假通过。异常测试覆盖 metadata、部分写入、PWAL，损坏通信组允许重建；TP>1 看 executor/RPC 后续行为。

### P8. P1/P2 并行要受优先级资源约束

位置：第 393、407 行。

可保留“P0后并行”，但补充：P1优先取得共享owner和硬件资源；P2只利用独立资源推进，不延误P1。明确每个P1里程碑选哪类draft/MTP方法，以及target-only、draft-only、共享权重和失败隔离的范围。

### P9. C3 和最终验收按优先级拆开

第 313、331 行的 C3 建议拆成 C3a（BF16 KPool/hooks/接口，P0）和 C3b（FP8 配对/反量化，P2）；P0 只要求支持的 BF16 配置不进入未适配量化分支。否则量化会重新进入 P0 关键路径。

第 426 行“capture 次数不变”只适用于图模式；eager 应验“不发生意外 capture”。第 430 行将 K4 拆为 P1 草稿验收、K5 为 P2 逐 scheme 验收，避免再次合并两级优先级。

## 3. 状态、格式和措辞优化

1. **补基线与信息截点。** 文章顶部应列两仓完整 SHA、源码检查日期、PR状态查询日期、无真机执行说明。密集源码行号只对该 pin 有效。
2. **#66487 状态已变化。** 第 271 行写 open；2026-11-18 API 已显示 merged，时间为北京时间 09:35:50。不过当前本地 pin 仍是 11-17 的提交，未包含该后续修复。应分成“上游已合入”“当前pin是否包含”两列，避免混淆。其余本次查询的 #65799/#66002/#65353 仍 open，#66666/#68032 仍 draft；这些也是查询时点状态，不是永久结论。
3. **“16项”限定统计集合。** 第 265、296 行写“此前选定清单中16项已合入，当前pin确认包含”，不要暗示穷尽整个RFC的全部相关PR；后续已新增合入项。
4. **“天然可用”改“公共实现可复用，接入待验”。** U1/U3的纯torch/storage语义不能替代NPU internal-format验证；U6还受Ascend自有模型loader影响；U9应按能力和可达性分解。第291行的“A3/A4”与实际hook清单编号不对应，可改为“C1–C3”。
5. **状态列不要只用✔/✘。** 第301行定义“✔已有脚手架”，读者容易看成已通过；建议分“已有实现/部分适配/静态缺口/候选设计/待NPU验证”，再单列运行验收状态。现有e2e有覆盖缺口不等于全部已有验证不存在。
6. **“零成本重解释”改“视图本身不搬运权重字节”。** begin仍有遍历、检查、属性恢复和padding清零，不能宣称整个begin零成本。第122/124行共同修改。
7. **避免普遍化PWAL和layerwise问题。** 第18–22、177行建议使用“部分路径会产生……”“PWAL可能非幂等，需按checkpoint输入重跑”，不要描述成所有layerwise必错或所有PWAL都非幂等。
8. **eager也属于目标。** 1.1可写“eager要求新状态与冷B一致；图模式在此基础上额外要求原图有效”，避免开篇给人只有图模型才适用的印象。
9. **术语收敛。** “按直连方式冷启动”改“在初始化模型前启用direct并从checkpoint冷加载”；“主流NCCL broadcast”改“trainer侧broadcast”；用“运行态原位更新”取代没有限定的“零拷贝”。
10. **长引用块与宽表简化。** 第128行承担了start、plan、资源借用、循环、broadcast和finish六件事，建议用T5的树形链路替换；详细行号放代码索引表，保留正文因果关系。

## 4. 建议补充的短章节

在2.9前后增加“普通checkpoint/磁盘/online quant/load-plan/draft五条入口的差异”，各用一行说明。特别是最新online quant：建层时选择匹配checkpoint method并包装loader，BF16在到达时量化为values/scales后写入；不是把BF16直接view进容量较小的量化storage。ND staging仍是未确认的NPU候选，不是通用online路径。

在3.2前增加“实现事实/适配方案/验证结果”的标记规则，配合状态列贯穿全文。已有上游代码、已有Ascend代码、建议新增helper、需要探针的NZ方案、历史文档声称的测试结果，分别标注。

推荐修改顺序：先改T1–T8等实现硬错与链路，随后消除P0/sleep和W0/W1依赖矛盾，最后统一基线、状态列和长段措辞。无需重写整份结构。

PR状态来源：[当前 #66487](https://github.com/vllm-project/vllm/pull/66487)、[完整性 #65799](https://github.com/vllm-project/vllm/pull/65799)。本文源码结论以本地固定pin为准，未据上游新合入自动改写当前源码事实。
