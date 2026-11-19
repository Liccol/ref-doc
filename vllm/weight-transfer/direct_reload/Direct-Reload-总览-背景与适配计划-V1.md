# Direct Reload（RFC #54477）总览：实现背景、代码逻辑、适配优先级与计划

---

> 修订日期：2026-11-18。源码核对基线：vLLM `3d836e13ea80247f0f7a2a4133a63e32d379fa0f`；vllm-ascend `b53996aad5d959f0b4511b4b87aadb2eb425b045`。本版依据固定基线静态核对，未进行 NPU 真机验收；正文源码行号仅对应上述基线。PR 状态截点为 2026-11-18，不能据此推定本地基线已包含后续合入项。

本文区分三类结论：**代码事实**（固定基线已有实现）、**适配方案**（拟开发或待探针）、**验证要求**（通过后才能声明支持）。已有代码不等于 NPU 已验。

## 1. 实现背景

### 1.1 要解决的问题

RL 训练每一轮产出新权重 B，推理侧正在服务的模型必须从 A 换成 B。更新完成后：

1. 状态与"独立冷加载 B"一致（数值等价）；
2. 图模式在数值等价的基础上，还要求原捕获图仍有效（图引用地址不变，不靠重捕获通过）；eager 不要求已有计算图。

checkpoint 与内核运行态经常不同（转置/打包、融合 QKV、MoE expert list、scale 倒数、MLA 派生矩阵），因此仅覆盖几个 Parameter 不能完成更新。

### 1.2 旧做法（layerwise reload）的局限

layerwise 是**逐层重建**：临时 materialize 参数 → 重放 loader → 该层 PWAL → `copy_` 回旧 kernel tensors。需要重点审查的成本和风险：

- 每轮都要新建临时参数，重建过程重跑后处理，成本高；
- 部分派生对象（普通属性/closure/expert list）可能仍指向被丢弃的临时参数，形成不报错但输出错的风险；
- 层完成与否依赖 loader 元素计数；若提前 finalize 可能静默数值错位，乱序到达时多个半层输入缓存也可能抬高内存峰值。这些是特定路径需验证的风险，不表示所有 layerwise 配置均有错误。

### 1.3 RFC 的演进与最终结论

- Rev 1 想用"PWAL 分类、更新时跳过一部分后处理"省成本；**Rev 2 放弃**——跳过 PWAL 会让 MLA 等派生数据继续读旧权重。
- Rev 2 结论：**更新完成的模型必须等价于"用新 checkpoint 冷启动"，同时保留已捕获图所引用的地址。**

### 1.4 与 layerwise 的核心区别

> 核心区别是重载过程中权重放在哪里、由谁负责保住图引用。**两者都以复用原图为目标。**

| 维度 | Layerwise | Direct |
| --- | --- | --- |
| checkpoint 写入位置 | 临时 materialized 参数，最后统一拷回 | 原参数 storage 的 checkpoint 视图，直接覆盖字节 |
| 保址责任 | `_copy_and_restore_kernel_tensors` 集中回拷 registered 参数/buffer | 每条 PWAL 与派生更新路径自己保址 |
| PWAL 时机 | 一层收齐就处理，特殊层延后 | 整轮 finish 完整重跑 |
| 完成判断 | 包装 loader + 元素计数 | 无逐层计数，完整性靠调用方与测试 |
| 部分更新 | 未发送层可恢复旧 kernel tensors（有限） | 非幂等/变布局层重跑 PWAL，漏发可能静默出错 |
| 失败恢复 | 两者都**没有**模型级原子更新和自动回滚 | finish 失败消费 begun 标记，不能重试同一 finish |
| 默认状态 | 本地默认 `VLLM_RELOAD_MODE=layerwise` | 显式开启，按进程固定 |

Direct 消除的是逐层 rematerialize + 通用回拷；**不等于整个更新零分配零拷贝**。迁移到 direct 后，NPU 自有 scheme 必须自己维护 storage、物理 format 和消费者引用——这是 Ascend 适配工作量集中的原因。

### 1.5 上游演进时间线

| 时间 | 变化 | 影响 |
| --- | --- | --- |
| 10-27 Rev 2 | 放弃选择性跳过 PWAL，转完整重跑 + 派生保址 | 按最终机制审查所有 NPU PWAL |
| 10-28~11-06 | 参数保留、direct core、派生刷新、无参 hooks、offset、统一 finish、padding、sleep2 陆续合入 | 形成参数保留、统一收尾、sleep/传输的基础能力 |
| 11-16 前后 | refit(#65347)、packed 顺序(#65487)、indexer/MLA 独立加载(#66118)、online quant(#65802) 四项新合入 | 新增 refit（权重热更新后与独立冷加载结果一致）的验证体系；packed 修复纳入通信适配；online quant 单列 |

---

## 2. 代码逻辑（direct reload 主链路）

本章先描述普通 checkpoint-format、target 模型、未启用 load plan/online quant 的上游路径；磁盘、plan、online quant 和 draft 的差异见 2.10。Ascend 的对应接入仍需第 3 章适配与验收。

### 2.1 会话状态机与 API

传输会话采用三段 API：`start_weight_update → update_weights × N → finish_weight_update`。HCCL/NPU IPC 等数据面负责传输，模型 loader 与 PWAL 负责加载和转换。磁盘 `reload_weights` 也复用 direct 生命周期，但不必调用这三段传输 API。

```text
READY（可推理；图模式下已有捕获图）
  --start/begin--> UPDATING（加载视图已恢复，禁止推理）
  --update_weights × N--> UPDATING
  --finish/PWAL 成功--> READY（运行态恢复）

前置检查拒绝：可能尚未改写模型，按会话和模型实际状态处理
begin 改写后、load 或 finish 失败：保持不服务，无自动回滚
  → 结束旧会话并显式发起新的完整更新；不可恢复则重建
```

关键语义：

- **begin 改写后到 finish 成功之间不能推理**：模型可能是 checkpoint 布局、旧字节或部分新数据的混合状态；
- **没有自动回滚**：不保存完整快照；进入 `finish_direct_reload` 后 begun 标记被消费，不能重试同一个 finish；
- 可恢复的数据/后处理失败，应清理旧会话的 active/目标状态，再完整更新；communicator、stream 或 rank 故障可能需要重建引擎/进程；
- incomplete 是当前进程内“已开始但未成功结束”的状态，不是跨进程的持久恢复记录。

### 2.2 冷启动：在 PWAL 之前记录 checkpoint 布局

```
GPUWorker.load_model                                    # v1/worker/gpu_worker.py:629
  └─ model_runner.load_model
      └─ BaseModelLoader.load_model                     # model_loader/base_loader.py:76
          ├─ create_model：建参数和 quant method         # base_loader.py:50（initialize_model）
          ├─ load_weights：原 loader 冷加载 A
          │     loader 级：base_loader.py:101 self.load_weights(...)
          │       → 实现于 default_loader.py:523（DefaultModelLoader）
          │       → default_loader.py:536 调 model.load_weights(...)
          │     模型级：模型/子模块的 load_weights / weight_loader（AutoWeightsLoader 路由）
          └─ process_weights_after_loading(model, config, device)
              ├─ record_checkpoint_layout（direct 开启时）     ← 记录在 PWAL 之前
              ├─ 逐层 quant_method.process_weights_after_loading(module)
              └─ finalize_model_weights：module hooks
  └─ 配置了 weight_transfer_config：WeightTransferEngineFactory.create_engine
后续：profiling / warmup / 图捕获（图记住 kernel 实际读取的地址）
```

record_checkpoint_layout记录的四组元数据（位于 `reload/direct.py:116::record_checkpoint_layout`）：

| 元数据 | 内容 | 服务环节 | 少了它会怎样 |
| --- | --- | --- | --- |
| `model._ckpt_owner`（`:127`） | 模型张量所有权哨兵 | begin 拒绝共享张量、padding 只清自己的（`:183-189`、`:230`） | draft 复用 target 的 lm_head 时写坏数据 |
| `tensor._ckpt`（`:130-132`） | owner + (shape, stride, dtype, storage offset)（`:68`）+ loader 属性（`_LOADER_ATTRS`，`:44`）+ 当时 storage 基址 | begin 重建 checkpoint 视图（`:190-201`）、恢复 loader 属性（`:224-226`）、storage 搬移后修正 offset 与容量校验（`:192-199`） | begin 不知道恢复成什么布局；loader 按错误打包维度分片；storage 被搬过时 offset 用错（#65281） |
| `model._ckpt_param_names`（`:141-145`） | checkpoint 参数名集合，含别名（`remove_duplicate=False`） | begin 的 `_check_params_kept`（`:148-162`）主要检测同名加载 Parameter 是否被替换（不覆盖所有删除/改名） | PWAL 替换加载对象后发现不了，checkpoint 字节写进布局不同的新参数 |
| `model._ckpt_padding_fill`（`:134-139`） | begin 时需清零的 MoE padding 参数清单（按 name 记，PWAL 前收集） | begin 对 owned 参数 `zero_()`（`:228-231`） | padding 残留旧值/垃圾值，kernel 按整块读到脏数据 |

四条设计要点：记录必须在 PWAL 之前（之后张量已不是 checkpoint 布局）；`_ckpt` 挂在加载张量自己身上（PWAL 应保留加载身份；替换对象可能丢失记录）；按对象去重、按名字记全；**图捕获的是地址**，不是参数名。

### 2.3 begin：校验与视图恢复（`begin_direct_reload`）

> 2.2 与本节之间隔了：PWAL 完整执行 → profiling/warmup/图捕获 → 服务期（用 A 推理）→ 调用方 pause 排空请求 →（可选 sleep2：备份/释放/训练产出 B/wake weights）→ `start_weight_update`（检查 engine/会话，标记 incomplete，调用`lifecycle.start_reload` 进入 `begin_direct_reload` ）。

顺序：检查冷启动记录存在 → 检查参数加载身份（非完整名字集合校验） → 提前计算并校验所有 checkpoint views → 设置 begun 标记 → `t.data = view` 切换加载视图 → 恢复 loader 属性 → 清零选定 padding。

四类拒绝及正确做法：

| 拒绝 | 含义 | 正确做法 |
| --- | --- | --- |
| 无记录 | 该模型没走 direct 冷启动 | 初始化模型前开启 direct，再从 checkpoint 冷加载 |
| 参数对象被换 | 同名加载 Parameter 被替换，丢失记录 | 改用 `replace_parameter` / `retire_parameter` |
| 共享张量布局被改 | draft 复用 target 参数且不重发 | 让共享方各自拥有，或在 target 侧更新 |
| 容量不足 | 当前 storage 无法容纳记录的 checkpoint 视图 | 拒绝该配置，或采用经探针验证的加载源方案；ND staging 是候选 |

容量判定用 stride 计算真实跨度（而非 numel），叠加 offset 修正。checkpoint 视图本身不搬运权重字节，也不做逆变换；整个 begin 仍有遍历、校验、属性恢复和 padding 清零。视图恢复不等于运行态恢复，因此期间禁止推理。`_check_params_kept` 没有用原始名字集合差检测全部删除/改名，仍需加载台账和专项测试补齐。

**checkpoint views 的构造细节**（`direct.py:170::_checkpoint_views` + `utils.py:113::storage_view`）：对每个带 `_ckpt`、且当前布局≠记录布局的张量——storage 被搬走则用当前 `storage_offset()`（`:192-193`）；按 stride 算跨度做容量校验，`need > storage.nbytes()` 报错（`:194-199`，拒绝当前 storage 容量不足；逻辑 shape 变小不必然表示 storage 变小）；最后 `torch.empty(0).set_(storage, offset, shape, stride)` 零拷贝建视图，begin 里 `t.data = view`。视图不搬权重字节，也不是逆变换；已知 checkpoint views 在改写前统一校验。finish 中 `utils.py:126::_in_own_storage` 尽量把结果写回旧 storage，但容量、device、对齐等条件不满足时可能返回新张量，随后重绑 `.data`。公共工具并非严格保址拒绝器，Ascend 必须补足支持限制和图引用验收。

### 2.4 load：原 loader 原地写入

外层调用方先 pause/drain，再调用 trainer 的 `engine.send_weights()`（`nccl_engine.py:540`）。该方法内部完成 start、数据传输和 finish；不应先手动 start 再调用它。

```text
caller pause/drain
  → trainer engine.send_weights()
      → client.start_weight_update()
          → worker/engine → lifecycle.start_reload → begin_direct_reload
          → engine 内：若启用 load plan，在 start_reload 后构建 plan
          → engine start 返回后：worker 维护 active，并按配置借用 KV 余量
      → 对每个 broadcaster source slice 调用 _send_update
          → 线程：client.update_weights(payload)，发送元数据 RPC
          ‖ trainer 线程：_broadcast(meta, src)，发送权重字节
          → worker 接收流等 caller 流，再接收并加载
      → client.finish_weight_update() → _post_send_sync()
  → 仅成功后 resume
```

`send_weights` 是仓库提供的 trainer 侧编排入口，不是唯一实现方式；若使用低层手动 API，调用方自己完成整套三段协议，不能重复 start。update RPC 与 broadcast 必须并发，不能先等待 RPC 返回再 broadcast。

- 普通 checkpoint 路径复用 `model.load_weights → weight_loader → TP/EP/融合路由`。调用方需完整覆盖每个必需的非幂等/变布局模块及会被清零的参数；不能据此推定任意部分更新安全，也不等于必须无条件发送全部冻结张量；
- 一次 update API/source slice 内可以形成多个 packed wire chunk；chunk 到齐只代表数据阶段进度，不触发逐层后处理；
- 通信 buffer、转换临时张量仍然存在。load plan 直写和 online quant loader 包装是额外分支，见 2.10。

**以 NCCL 为例：worker 侧接收细节**（每次 update RPC，`nccl_engine.py:292::receive_weights`）：

```text
POST /update_weights（NCCLWeightTransferUpdateInfo 元数据）
  → AsyncLLM.update_weights → collective_rpc
  → GPUWorker.update_weights
      → engine.update_weights → parse_update_info
          → receive_weights                             # nccl_engine.py:292
              1. 前置守卫：group 未初始化报错（:309）；src ∈ [0, rank_offset)（:314）
              2. 元数据复用：names 非空则缓存进 _last_update_info（按 (model,src)），
                 为空则复用上次同 src 元数据（:320-324，对应 trainer 的 reuse=True）
              3. 写入口选择（:341-343）：有 load plan（仅 target）→ load_planned 直写，
                 否则 model.load_weights；plan 路径先用 _check_manifest 校验本桶元数据
              4. 接收路径：
                 packed（:344-361）：(name,(shape,dtype)) 迭代器 →
                   packed_nccl_broadcast_consumer(..., post_unpack_func=load_weights,
                     buffer_size_bytes, num_buffers, device)  —— 按握手时 trainer 传来的
                     buffer 几何解包字节，解出即交给 load_weights 写入
                 unpacked（:362-373）：逐参数小循环 torch.empty → group.broadcast
                   (stream=current_stream) → load_weights([(name, weight)]) → del
                   —— 每收一个参数立刻写、立刻释放，不攒桶
```


接收路径要点：

1. unpacked 每参数分配临时张量；packed 合并传输、按配置轮转并发 slot/stream。当前上游 consumer 每次 receive 分配 packed tensor，unpack 默认返回 views，并通过 event/record_stream 维持顺序和存活，不保证固定几个底层 storage 循环复用。
2. Ascend 本地 HCCL helper 显式 byte clone-unpack；NPU IPC 又有自己的 handle/buffer 复用策略。三者分别核对，不能把 Ascend 行为写成上游统一行为。
3. 上游 unpacked broadcast 在 `torch.cuda.current_stream()`（`:370`）上；Ascend 需验证 caller 的 begin/padding 到接收/写入，以及接收完成到 finish 的跨流顺序。
4. load plan 构建失败（`:273–281`）只会清除相关 metadata 并抛错，不自动补发或修复。调用方必须保持暂停，显式发起新的完整更新；后续可走普通 `load_weights`，成功后才恢复服务。重新尝试规划需 re-init engine。

### 2.5 finish：完整重跑 PWAL（`finish_direct_reload`）

> 2.4 与本节之间：**"本轮数据已送完"由调用方保证**（direct core 不做整轮完整性检查，#65799 未合入）；接收结束时 engine 做 `accelerator.synchronize` 等待接收/转换流落定；随后 finish RPC 进入 `finish_direct_reload` 的前置三步——消费 begun 标记（拒绝重复 finish）→ 设备同步 → 清除相关 quant module 的一次性 PWAL guard（guard 不清，PWAL 会误以为已处理而跳过转换）——之后才执行下面的完整重跑。

```
finish_direct_reload
  ├─ 消费 begun 标记，拒绝重复 finish
  ├─ 设备同步：等待接收/其他流上的写入完成
  ├─ 清除相关 quant module 的一次性 PWAL guard
  └─ process_weights_after_loading 完整重跑
      ├─ 冷启动记录已存在，不再重复记录
      ├─ 各层完整 PWAL：转置/打包/量化后处理 → 按各方法实现写回（图保址需验证）
      └─ finalize_model_weights：module hooks（children-first）
  → 释放可回收 allocator 内存 → worker 结束会话、归还资源、重置目标
  → 成功时清除 incomplete 标记，通知 target 替换
```

- 部分 PWAL 非幂等（量化、融合、吸收等）；direct 从正确 checkpoint 输入完整重跑所需后处理，并验证结果与冷启动一致；
- **图模式要求**写回图正在引用的地址；公共工具可能重绑 storage，无法证明保址的配置必须在支持入口拒绝或另行设计，不能把该目标当成已有通用报错保证；
- 派生值（scale 倒数、MLA 派生、expert list、融合 gate）必须写回图正在读的那个对象。

**2.2 → 2.5 全链路时序（普通 checkpoint 路径）**：

```text
冷启动：record_checkpoint_layout → PWAL → warmup/图捕获 → 服务 A
更新轮：caller pause/drain → trainer send_weights
  start RPC → engine start_reload → begin（校验/views/loader 属性/padding）
            → 可选 plan 构建 → 返回 worker，维护会话和资源
  update RPC ‖ broadcast → 接收流等 caller 流 → loader 写入（可重复多次）
  调用方保证送完 → finish RPC → 消费 begun/同步/清一次性 guard
            → 完整 PWAL/hooks → 成功结束会话、归还资源、清 incomplete
  caller 仅成功后 resume；失败保持暂停并按状态修复或重建
```

### 2.6 PWAL 内部机制：后处理怎么逐层调度、转换结果怎么保住图地址

> 本节不在时间轴上，是 2.5 中"process_weights_after_loading 完整重跑"的**纵切面**：它内部每层怎么调度、每个转换结果怎么不弄丢图地址。2.1–2.5 是宏观链路，2.6 起为专题剖析。

`process_weights_after_loading` 是"调度入口 + 两大家族实现"：

- **公共调度入口**：`model_loader/utils.py:152`。流程：`maybe_retie_word_embeddings` →（direct 开启时）`record_checkpoint_layout` → 遍历 `named_modules`，对带 `quant_method` 的模块调 `quant_method.process_weights_after_loading(module)`（`utils.py:212`，逐层 quant pass）→ `finalize_model_weights`（`utils.py:224`）。冷启动 `base_loader.py:122` 与 direct 的 `finish_direct_reload`（`reload/direct.py`）共用此入口。
- **quant method 实现**（签名带 `layer`）：`layers/linear.py:208`、`layers/fused_moe/*`、`kernels/linear/scaled_mm/*`、`layers/quantization/*` 等数百处，由逐层 pass 调用。
- **module hook 实现**（无参 `self`）：`layers/attention/attention.py:631`、`mla_attention.py:1502`、`parameter.py:498` 等，由 hook 遍历调用。

统一调度（#57678/#65914，`weight_utils.py:1492::finalize_model_weights`）：

| 阶段 | 行为与边界 |
| --- | --- |
| quant-method pass | 公共入口遍历带 quant_method 的层，调用 method 的带 layer 后处理 |
| module-hook pass | `finalize_model_weights` 采用 children-first；共享 module 去重；根最后 |
| 遍历边界 | 带 quant_method 层的子树不递归，QuantizeMethodBase 实例跳过；相关所有者负责覆盖 |
| deferred attention | hook 调用时按需选择 device loading context，不是单独一轮 pass，也不是遍历跳过项 |

module hook 统一为 `process_weights_after_loading(self)`，所需 dtype 可在构造期保存；quant method 仍带 layer。只有公共遍历已覆盖的同一 hook 委托才删除重复转发；backend 非注册 Module、quantized 子树等仍可能需要显式委托，backend 自身带 dtype 的方法不必机械改签名。

公共工具（`model_executor/utils.py`，已有实现不等于严格保址保证）：

| 工具 | 当前用途 | 布局/容量变化时的边界 |
| --- | --- | --- |
| `replace_parameter` / `_in_own_storage` | 尽量复用旧 Parameter/storage 写回结果 | 条件不满足可能返回新张量、重绑 `.data`，不统一报错 |
| `retire_parameter` | 退休源参数，保留注册与数据、可再加载，kernel 枚举跳过 | 仍需核对消费者与加载身份 |
| `move_parameter` | 改名场景保留同一对象 | 删除/改名覆盖需专项验证 |
| `refill_derived` | 条件匹配时原位 copy 派生属性 | 不匹配可 setattr 换对象；调用方负责图契约 |
| `storage_view` | 在指定 storage 建 shape/stride/dtype/offset 视图 | 自身不做容量验证，由 `_checkpoint_views` 等调用方先检查 |

**Ascend 候选工具**：专家列表逐元素保址回填（含 E8M0 字节保真）、非持久 buffer 保址刷新可按命中路径补充公共 helper。此前方案使用的 `refill_expert_list`、`refill_derived_buffer` 不是当前两仓已有的通用函数名，不能作为现成 API 依赖。

### 2.7 三类典型失效：布局错、地址错、派生值错

1. **布局错**：begin 没把张量变回 checkpoint 布局 → loader 把 checkpoint 字节写进运行布局，转置/打包/NZ 整体错位；
2. **地址错**：后处理每轮新建张量重绑属性 → 图引用的地址被释放/复用，不报错但偶发乱码（最危险：shape 对、不崩、只在某些轮触发）；
3. **派生值错**：scale/倒数/MLA 派生/expert list/融合 gate 没刷新 → weight 新了输出还像旧权重。

统一修法：**记录布局 → begin 视图 → 原 loader 写入 → 后处理写回原地址**。

### 2.8 MoE padding 清零与 meta 张量初始化

- MoE 专家权重 pad 到对齐尺寸，loader 不写 padding 但 kernel 按整块读，更新后必须清零；
- selector 只在"维度对不上"时清零；per-expert 标量 `(E,)`、`(E,≤2)` 不是 padding 对象，混入会把真实数据清零；
- 清零擦掉 checkpoint 区域 → **清零与"必须完整重发"绑定**；
- `materialize_meta_tensor` 改为 `empty_strided(...).zero_()` 补 0。

### 2.9 特性使能：配置开关、使能方式与每轮更新入口

| 设置 | 负责什么 |
| --- | --- |
| `VLLM_RELOAD_MODE=direct` | 冷启动记录布局 + 更新走 direct 协议（需在启动前设置） |
| `--weight-transfer-config '{"backend":"nccl"}'` | Ascend 对应 `hccl` / `npu_ipc`；创建并绑定传输 engine |
| `init_weight_transfer_engine(init_info)` | trainer/worker rendezvous，建 communicator（不开始改写权重） |
| `VLLM_SERVER_DEV_MODE=1` | 注册 HTTP 的 pause/resume 与权重更新路由 |

推荐每轮更新由 trainer engine 编排：`pause → send_weights（start → 逐桶 update_weights+broadcast 并发 → finish）→ 成功后 resume`。`send_weights` 不自动 pause/resume，调用方负责；update RPC 与 broadcast 必须并发，不能先等 RPC 返回再 broadcast。

最短验证入口（无 trainer）：`VLLM_RELOAD_MODE=direct` + `LLM(model=A)` + `collective_rpc("reload_weights", kwargs={"weights_path": B, "is_checkpoint_format": True})`，走 `reload/inplace.py`，不建通信组。

配置检查（`config/vllm.py::_verify_direct_reload_compat`）当前仍拒绝：LoRA、weight offloading、torchao、`draft_lm_head_quantization`、sharded RDT 等。

### 2.10 不同入口的边界

| 路径 | 与上述普通 target checkpoint 路径的差异 |
| --- | --- |
| 普通 checkpoint + 传输 | 三段 API 编排，接收后复用模型 loader，再统一 finish |
| 磁盘 reload_weights | 复用 direct 生命周期，从磁盘加载，不建传输组、不调用外部三段传输 API |
| online quant | 建层时选择 checkpoint method 并包装 loader；BF16 到达后生成量化 values/scales 再写入，不是将 BF16 直接 view 进小容量量化 storage |
| load plan | 首次 start 内构建写入计划，后续可按计划直写并检查 manifest；Ascend 当前未接通，先拒绝 |
| draft | 有独立目标、start 和共享权重约束，不能直接套用 target-only 支持结论；列 P1 |

ND staging 是强制 NZ 等路径的候选适配设计，不是当前上游 online quant 的通用实现。

### 2.11 上游已合入集合与边界

此前选定清单中的 16 项相关 PR 已包含在当前本地基线，其中后期 4 项为 #65347 refit、#65487 packed 顺序、#66118 indexer/MLA、#65802 online quant。这个数字不穷尽 RFC 全部相关 PR。以下将最新上游状态与本地源码覆盖分开（2026-11-18 查询）：

| 项目 | 上游状态 | 当前基线/适配边界 |
| --- | --- | --- |
| [#65799 参数级完整性](https://github.com/vllm-project/vllm/pull/65799) | open | 本地无整轮完整性检查，依赖调用方与测试 |
| [#66002 能力声明/拒绝](https://github.com/vllm-project/vllm/pull/66002) | open | 支持模型需自行维护支持表与拒绝入口 |
| [#66487 finish 不重新 tie embeddings](https://github.com/vllm-project/vllm/pull/66487) | 已于 11-18 北京时间 09:35 合入 | 当前 11-17 基线未包含；选定模型需专测 lm_head 独立性，升级后重新核对 |
| [#65353 gpt-oss streamed TP](https://github.com/vllm-project/vllm/pull/65353) | open | 不声明相应模型支持 |
| [#66666](https://github.com/vllm-project/vllm/pull/66666)/[#68032](https://github.com/vllm-project/vllm/pull/68032) KV scale | draft/open | 不承诺任意 FP8 KV |
| pending-update 推理门禁 | RFC 后续项 | 受控发布由调用方严格暂停/排空，不能据 RFC 推定当前已有门禁 |


---

## 3. 适配点清单与场景优先级（2026-11-19 修订）

**按模型与传输场景递进交付。** 每个阶段明确完整模型、执行模式、传输配置和退出条件，以同一累计版本的端到端结果确认场景可用。

交付顺序为全层 Qwen3 dense/MoE 的 HCCL unpacked → 四条传输路径 → Qwen3.5 → 存量 GLM/DeepSeek/MiniMax/Qwen e2e 验证及修复，随后依次扩 MTP 和低精。第三章列出各项进入哪个阶段，第四章列出任务依赖和验收方法。

公共实现可作为基础复用。Ascend 自有 loader、覆写、internal format、设备/流和生命周期仍需按各阶段核对。

### 3.1 公共实现可复用清单

| # | 可复用能力 | 位置 | 接入边界 |
| --- | --- | --- | --- |
| U1 | record/begin/finish、checkpoint views | `model_executor/model_loader/reload/direct.py` | 公共 storage 逻辑可复用；NPU internal format、容量和 accelerator 同步待验证 |
| U2 | start_reload/finish_reload、active 标记 | `reload/lifecycle.py` | 公共模式分派可复用，NPUWorker 会话收尾另补 |
| U3 | replace/retire/move/refill_derived/storage_view | `model_executor/utils.py` | 已有 torch 工具；可能重绑对象，不代替图保址拒绝和 NPU 格式验证 |
| U4 | children-first hook、quant 子树边界 | `weight_utils.py::finalize_model_weights` | Ascend module/quant method 需正确接入，随具体模型与量化场景验证 |
| U5 | 冷加载→记录→PWAL 的公共顺序 | `base_loader.py`、`model_loader/utils.py` | 复用公共 loader 的路径可继承；自有 runner/loader 仍核对实际可达性 |
| U6 | 模型 loader、TP/EP/融合路由 | 模型定义、`default_loader.py` | 更新复用冷加载逻辑；Ascend 自有模型与 loader 需逐项验证 |
| U7 | HTTP/RPC、HTTPVLLMWeightSyncClient | AsyncLLM/protocol 层 | 公共路由可复用；worker 方法和异常语义需接通 |
| U8 | 通信生命周期、公共 update wrapper 和控制协议 | `distributed/weight_transfer/` | 仅未被覆写部分可继承；Ascend init/receive/packed/metadata/plan 分别适配，不能自动获得父类全部逻辑 |
| U9 | 此前选定集合的上游公共改动 | 当前 vLLM 基线 `3d836e13ea`，见 2.11 | 源码已包含，能力是否在 NPU 路径可达、兼容仍分项核对 |
| U10 | 独立冷 B 的 refit oracle 语义 | #65347 | 可复用测试语义，NPU fixture/device/graph 观察器需适配 |

### 3.2 各场景的适配内容

历史缺口来自本文固定基线，实施时对照 #22169、当前主线和后续补丁重新核对。已有实现和真机结果分别登记，支持范围绑定版本、模型、硬件和运行配置。

| 阶段 | 适配内容 | 进入后续阶段的条件 |
| --- | --- | --- |
| 1 · 全层 Qwen3 dense/MoE，HCCL unpacked | 核对真实 worker 会话、device/config、异常收尾、incomplete 与支持限制；保留 checkpoint 名称、参数身份、loader 属性和别名；验证 dense 的分片/融合与 MoE 的专家列表、padding、实际 ND/NZ；建立冷 B、原图、失败修复和资源检查 | 两个全层 BF16 模型各自 eager/图通过，共 4 格。第二轮 begin 仍可正确加载，部分失败后可完整修复 |
| 2 · HCCL/IPC × unpacked/packed | 接续真实 HCCL unpacked，适配 HCCL packed、IPC unpacked 和 IPC packed。核对打包几何、metadata、多 chunk 顺序、buffer 所有权、device/UUID、handle 导入释放和连续更新 | 每个模型 8 格，共 16 格在同一累计版本执行；各通道都有冷 B、输出、图或 eager、失败与资源证据 |
| 3 · Qwen3.5 | 先复用缩层用例定位，再验完整型号；补 mixed attention、GDN、融合/专家派生状态和实际 hook 路径 | 先 HCCL unpacked eager/图，再扩其余通道。公共实现变化后回归 Qwen3 16 格 |
| 4 · 存量 GLM/DeepSeek/MiniMax/Qwen e2e | 逐用例盘点精度、入口与 Layerwise 状态。GLM 检查 KDA/SFA/KPool 和 Indexer，DeepSeek 检查 sink、MLA/wo_a 与专家派生状态，Qwen 检查新增混合/融合路径；MiniMax 按实际用例的精度归入 BF16 或后续低精验收 | 每个支持行有真实入口、全层范围、独立冷 B 和原图结果。已 skip、offload 或量化组合记录阻塞与后续依赖 |
| 5 · MTP/投机推理 | 从已验 BF16 target 验证 draft 派生刷新、独立更新、共享 embedding/lm_head 和失败隔离 | 选定方法的 target/draft 更新结果、两侧状态与恢复行为通过 |
| 6 · 低精 | 预量化逐 scheme 检查 weight/scale/pack/offset、转换 guard、host scalar 和图引用；online quant 单验 checkpoint method/loader 映射与 values/scales | 每个 scheme 有同格式冷 B、实际内核和通道结果 |
| 配套 · sleep/offload | 检查 allocation/tag、纯参数与混合块、备份、分段 wake、原图生命周期和资源回收 | 对已稳定模型/通道逐配置增加支持，P0 首跑保持 no-sleep |

完整性、会话与拒绝入口随阶段 1 首个可用增量核对；load plan 等未接通能力在入口拒绝，后续有需求时单列场景。

### 3.3 优先级与阶段验收

| 优先级/阶段 | 交付范围 | 最低验收 |
| --- | --- | --- |
| P0 阶段 1 | HCCL unpacked，全层 Qwen3 dense/MoE，BF16 eager/图 | 两模型各两格，共 4 格。真实 trainer 全量发送，A→A→B→A、独立冷 B、图 replay、失败恢复与资源证据，记 S0 |
| P0 阶段 2 | 两个全层 Qwen3 模型扩到 HCCL/IPC × unpacked/packed | 每模型 8 格，共 16 格，在同一累计版本通过。保留阶段 1 四格回归，记 S1 |
| P0 阶段 3 | Qwen3.5 的全层指定型号、mixed attention 和实际派生状态 | 先 HCCL unpacked eager/图，再扩其他三条传输路径。缩层用例用于定位，型号支持以全层证据登记 |
| P0 阶段 4 | GLM/DeepSeek/MiniMax/Qwen 存量权重更新 e2e 验证与修复 | 按已有可执行用例、精度、硬件和入口登记 Layerwise/Direct。BF16 新分支逐模型适配，skip 与量化/offload 组合明确后续依赖 |
| P1 阶段 5 | MTP 与投机推理 | 选定方法的 target/draft 刷新、独立更新/共享权重和失败隔离 |
| P2 阶段 6 | 预量化、online quant 等低精 | 逐 scheme 验 checkpoint、转换、scale/派生值、原图与输出；共享资源优先保证 P0/P1 |
| 配套 | sleep2/CaMem、offload 等 | P0 首跑 no-sleep，已有稳定组合可开展附加测试，按声明范围独立交付 |

BF16 模型实际需要的 NZ、融合/转置、padding 和派生状态保址，随该模型所在阶段完成。阶段 1 只要求所选 Qwen3 配置可达的分支，后续模型的专有实现可并行准备。完整性、会话和支持限制在首个可用增量核对，复杂模型及两条通信线可在各自依赖就绪后提前开发。

阶段 2 的基础验收矩阵如下。每一格绑定 checkpoint、两仓 SHA、硬件、TP/EP 和具体图模式；A3/A5 分别登记实际支持证据。

| 全层 BF16 模型/执行模式 | HCCL unpacked | HCCL packed | IPC unpacked | IPC packed |
| --- | --- | --- | --- | --- |
| Qwen3 dense eager | 阶段 1 必验 | 阶段 2 必验 | 阶段 2 必验 | 阶段 2 必验 |
| Qwen3 dense 图 | 阶段 1 必验 | 阶段 2 必验 | 阶段 2 必验 | 阶段 2 必验 |
| Qwen3 MoE eager | 阶段 1 必验 | 阶段 2 必验 | 阶段 2 必验 | 阶段 2 必验 |
| Qwen3 MoE 图 | 阶段 1 必验 | 阶段 2 必验 | 阶段 2 必验 | 阶段 2 必验 |

---

## 4. 适配计划

### 4.1 按模型与传输场景递进交付（2026-11-19）

交付顺序采用：**HCCL unpacked 全层 Qwen3 dense/MoE → HCCL/IPC × unpacked/packed → Qwen3.5 → 存量 GLM/DeepSeek/MiniMax/Qwen 模型 e2e 回归与修复 → MTP → 低精**。前两个阶段先形成稳定的 BF16 通用底座。Qwen3.5 单列混合 attention 的模型适配，后续模型族按实际用例逐项扩展。

“全层”指所选 checkpoint 保留原始层数和模型结构，包括全部专家。缩层、缩专家和 toy 模型可用于定位，最终验收绑定全层配置。首批候选建议 dense 使用 Qwen3-0.6B、MoE 使用 Qwen3-30B-A3B 的 BF16 checkpoint，实际 checkpoint/revision 及可运行的 TP/EP 由开工资源评估确定。每项结果记录模型结构和参数 dtype。

首批采用 target-only、未量化权重/KV、no-sleep；图模式先明确一种可执行配置，建议从 FULL_DECODE_ONLY 起步，登记 capture sizes 和实际 replay。eager 与图模式分别验收。A3/A5 按实际声明的硬件分别留证，设备资源决定两者先后。IPC no-sleep 在开工时测量 trainer 与推理进程的显存峰值，必要时调整 TP 和部署，sleep2 配置另行登记。

| 阶段 | 核心交付 | 退出条件 |
| --- | --- | --- |
| 0 准备 | 对齐双仓与 #22169、首批模型、公共测试和支持限制 | 固定 SHA/checkpoint、设备/流约定、独立冷 B 及运行命令，足以并行开发 |
| 1 HCCL unpacked 基础 | 全层 Qwen3 dense 与 MoE 的真实权重更新 | 2 模型 × eager/图共 4 格在同一累计版本通过，含重复更新与失败修复，记 S0 |
| 2 传输矩阵 | 在阶段 1 两个模型上补 HCCL packed、IPC unpacked、IPC packed | 每个模型 8 格，共 16 格真实执行，记 S1；同硬件/部署范围内保持相同 checkpoint 和参考数据 |
| 3 Qwen3.5 | 复用底座，补 mixed attention、GDN 及 MoE 等实际后处理路径 | 先真实 HCCL unpacked eager/图，再复跑四条传输路径；全层型号逐项验收 |
| 4 存量模型 e2e | GLM/DeepSeek/MiniMax/Qwen 的权重更新用例盘点、参数化和 bug 修复 | 对每个已有可执行 Layerwise 权重更新用例登记 Direct 结果与范围，阻塞项有定位和责任人 |
| 5 MTP | 从已通过的 BF16 target 扩展 draft 刷新、独立更新与共享权重 | 选定投机方法完成两侧状态、输出和失败隔离验收，记 S2 |
| 6 低精 | 业务优先的一种预量化 scheme，再扩其他格式与 online quant | 每个 scheme 单独证明转换、scale、图引用和输出，记 S3 |
| 配套 sleep/offload | 可并行探针与资源设计，在稳定配置上补整轮交换 | 按承诺支持的具体配置验收分段 wake、状态恢复、原图和资源回收 |

#### 4.1.1 第一阶段：HCCL unpacked 全层 Qwen3 dense/MoE

| 子任务 | 实现/核对内容 | 依赖和验收 |
| --- | --- | --- |
| 1.1 固定候选与部署 | 核对 #22169 可复用改动和当前 worker 继承/覆写；固定 dense/MoE 的完整模型、TP、硬件与运行 kernel | 无前置。全层 BF16 冷 A/B 都可启动并推理，记录内存预算和参考结果 |
| 1.2 公共测试参数化 | case 支持模型、reload_mode、execution mode、backend、packing、sleep。用已有 fixture 加完整 Qwen3 用例，保留原 Layerwise 回归 | 与会话/通信开发并行。能明确收集本阶段 4 个 Direct cell，输出命令与配置 |
| 1.3 Worker 会话 | 正确 device/config、start/update/finish、active/incomplete、目标重置、支持范围和异常传播 | 真实入口验证重复 start、无 start update/finish、失败后新完整更新；支持限制随首个增量生效 |
| 1.4 Dense 加载闭环 | 原模型 loader、QKV/TP 分片、embedding/lm_head 别名、loader 属性和重复 PWAL | dense eager 先跑，逐参数/派生值与冷 B 对照。原始 checkpoint 为全层 |
| 1.5 HCCL unpacked 数据面 | trainer/worker rendezvous、rank/device、逐 tensor broadcast 与 update RPC 并发、begin/padding→receive/load 顺序、finish 和清理 | 与 1.3/1.4 并行实现后汇合。真实 trainer 发送 checkpoint 张量，全模型全部所需数据跨进程到达 |
| 1.6 Dense 原图与连续更新 | 同引擎执行 A→A→B→A，保留旧 storage，观察具名消费者、图对象/capture/replay 和输出 | 1.3–1.5 eager 通过后增加图格。B 使用不同的完整 checkpoint 或固定可复现变换，独立冷 B 使用同一份数据 |
| 1.7 MoE 加载与运行态 | 全层/全部专家、w13/w2、expert 列表、TP/EP 路由、padding、融合算子、实际 ND/NZ。核对第二轮 begin 的加载源和格式 | 基础链路复用，模型组件可提前做。先 MoE eager，再原图；FUSED_MC2 等实际选中 NZ 时保留本阶段适配 |
| 1.8 端到端失败与修复 | 前置拒绝、部分写入、漏发一个必需 tensor、PWAL 失败；暂停状态、metadata 变化、完整重发和多 rank RPC 后续行为 | 在可运行模型上逐项注入。需要完整性检查的分支先对接所选基线实现；损坏通信组允许重建 |
| 1.9 S0 验收 | dense/MoE 各 eager/图；A→A→B→A、冷 B、原图、HCCL 真传输、峰值/稳态内存、更新时延 | 4 个 Direct cell 在同一累计版本通过，记录对应 Layerwise 结果及未支持配置 |

第一阶段的临时磁盘/inplace smoke 用于区分加载与通信问题，可并行运行。S0 需要真实 HCCL unpacked 的全层模型证据。图场景同时检查模型状态和实际生成，恢复服务发生在成功 finish 后。每个阶段将实现增量与测试设施合入同一累计版本后验收。

#### 4.1.2 第二阶段：HCCL/IPC × unpacked/packed

| 子任务 | 实现/核对内容 | 依赖和验收 |
| --- | --- | --- |
| 2.1 固定协议与打包几何 | dtype/shape/name、buffer size/count、最大 tensor 与 chunk 边界、metadata 复用及设备/流语义 | 开工即可做，对齐 trainer 与 worker 的可达协议。保持现有支持输入兼容 |
| 2.2 HCCL packed | 复用阶段 1 会话与模型，核对 producer/consumer 顺序、解包结果是否共享 buffer、load 完成前的存活和下次覆盖 | 先 dense eager，稳定后 dense 图/MoE。从小 UT 到真实通信逐层定位，按证据解除对应 packed skip |
| 2.3 IPC unpacked | 逻辑/物理 device、UUID/handle 映射、trainer 与 receiver 张量生命周期、同步与释放 | 可与 2.2 并行。dense eager→dense 图→MoE，使用实际 IPC handle 跨进程传输 |
| 2.4 IPC packed | packed export 的导入次数、view/clone 所有权、buffer 复用和跨轮 handle 生命周期 | 依赖 2.1 和 IPC 基础通道，复用公共 packed 工具。dense/MoE eager/图逐项验 |
| 2.5 多 chunk 压力 | 整批、小 chunk、跨层/跨专家边界、尾 chunk、连续同 dtype/混合 dtype；每轮更新不同权重，制造通信与 load 的交叠 | 配置在已支持的几何范围内，超界输入显式拒绝。验证缓冲覆写前消费完成、输出仍与冷 B 一致 |
| 2.6 异常和重复会话 | 不匹配 metadata、load/PWAL 失败、部分传输；异常传播、handle/stream 清理、下一轮完整 metadata 与数据重发 | 各后端保留最小可复现失败。测试设超时，避免残留进程/通信组影响后续；通信故障按实际能力选择重建 |
| 2.7 累计矩阵与资源 | 同一累计版本跑两个全层模型全部 16 格，检查连续更新后内存、handle 和进程资源 | 单元格有真传输、冷 B、eager 或原图证据。所有 skipped/not collected 标待验，图模式/TP/硬件另列配置 |
| 2.8 S1 发布 | 支持表、复现命令、已知限制和 Layerwise 回归 | 对外支持范围与 16 格实际配置一致，原有已支持路径回归通过 |

第二阶段交付矩阵如下，每格均为真实传输。按硬件和选定并行部署分别登记，不隐含所有 TP/EP 或图模式。

| 全层 BF16 模型/执行模式 | HCCL unpacked | HCCL packed | IPC unpacked | IPC packed |
| --- | --- | --- | --- | --- |
| Qwen3 dense eager | 阶段 1 | 阶段 2 | 阶段 2 | 阶段 2 |
| Qwen3 dense 图 | 阶段 1 | 阶段 2 | 阶段 2 | 阶段 2 |
| Qwen3 MoE eager | 阶段 1 | 阶段 2 | 阶段 2 | 阶段 2 |
| Qwen3 MoE 图 | 阶段 1 | 阶段 2 | 阶段 2 | 阶段 2 |

#### 4.1.3 第三、四阶段：Qwen3.5 与存量模型扩散

Qwen3.5 先复用已有缩层用例检查 GDN/mixed attention/hook，再用选定完整 checkpoint 做全层 Direct 验证。第三阶段开始可选 Qwen3.5 dense 或 MoE 中一个确定型号首跑，随后扩另一个，型号和顺序在资源评估时登记。首次集成 HCCL unpacked，再应用第二阶段的其余通道。公共模型/通信代码变化后回归 Qwen3 16 格。

存量回归先按“权重更新 e2e”而非仅按模型名盘点。GLM 核对 KDA/SFA/KPool 与 Indexer，DeepSeek 核对 sink、MLA/wo_a 和专家派生状态，Qwen 核对已有混合 attention/融合路径；MiniMax 先查仓内实际用例和精度，存在 BF16 可执行配置时加入 BF16 队列，已有量化用例归低精专项。每项记录现有入口、Layerwise 能否执行、Direct 新增 cell、缺口和对应 PR。已 skip 的用例先复现阻塞，再决定修复或声明范围；offload/MTP 等组合分别登记依赖。

每个子任务可形成独立可验证增量。已经在 #22169 中实现的部分按最终累计版本复用，实际新 PR 聚焦剩余差异。发布前在同一版本复跑对应阶段矩阵，并保留此前已通过场景的回归。

### 4.2 开工前确认的场景条件

| 条件 | 最低内容 |
| --- | --- |
| 模型与数据 | 两仓 SHA、全层 checkpoint/revision、参数 dtype、可区分的 A/B、独立冷 B 及硬件/TP/EP |
| 更新会话 | start/load/finish 的实际入口、active/incomplete、失败后保持暂停与完整重发、支持范围和拒绝位置 |
| 设备与流 | begin/padding→receive/write→finish/PWAL 的顺序、buffer/handle 生命周期及失败释放 |
| 模型与原图 | 可写 checkpoint 源、实际 ND/NZ、派生消费者、PWAL/hook 顺序、图模式/capture sizes 与 replay |
| 验收数据 | 阶段 1 四格、阶段 2 累计 16 格的命令、结果字段、Layerwise 对照和原始日志 |

### 4.3 验收口径

发布支持声明需满足以下证据，eager 与图模式分别适用：

| 证据 | 最低要求 |
| --- | --- |
| 数值/状态 | A→A、A→B→A、独立冷 B oracle；参数/buffer/expert list/scale 与实际输出；确定性转换尽量 bitwise |
| 原图/存储 | 图场景检查具名消费者对象/alias/storage/data_ptr/offset/layout/物理 format，证明实际 replay、无 eager fallback、无额外 capture；保留旧 storage 引用，防释放后同址分配假通过。eager 不要求已有图，检查无意外 capture |
| 入口/故障 | 真实 worker start/update/finish；覆盖前置拒绝、metadata、部分写入、PWAL 故障；失败后保持暂停，可恢复时新完整更新，通信组损坏允许重建；TP>1 观察 executor/RPC 后续行为 |
| P0 资源/组合 | no-sleep 基线下验证 padding、跨流、声明支持的 TP/多卡、NPU/CPU 峰值与稳态内存、更新时延 |
| sleep2 附加 | 仅对声明支持 sleep2 的组合要求：分 tag wake、纯参数/混合 allocation、非参数状态保留及原图生命周期；不作为全部 P0 组合默认门槛 |

按阶段确认交付：阶段 1 完成 4 格，阶段 2 在同一累计版本完成 16 格；随后按 Qwen3.5、存量模型、MTP 和低精的具体支持配置递进验收。每条结果都绑定完整累计 tree 和对应原始日志。

每条结果绑定：两仓 SHA、累计 tree、模型/checkpoint 与 A/B 标识、硬件/CANN/torch_npu、dtype/实际 quant method、TP/EP、execution mode、图模式/capture sizes、backend、packed、sleep、load plan、命令和日志。八组合通过只覆盖支持表中明确登记的模型与配置，不泛化到所有 BF16 模型。

台账按阶段和场景记录实现增量、来源 PR、模型/算子、支持配置与运行结果。分别登记实现状态和 NPU/端到端验收状态，区分已适配、已验、不适用、明确拒绝和待验证。


### 4.4 现有 PR 与 Layerwise 回归台账（2026-11-19）

本节记录 4.1 交付计划使用的 PR 和存量回归。源码对照使用 Ascend 上游 `main@ed58d21a18922862432ce48574270dcc8438779c`，与本文开头用于代码走读的本地基线分别记录。本节测试结果来自 PR 报告，真机验收需在最终累计版本补齐。

**PR 定位。** [Ascend #22169 变更](https://github.com/vllm-project/vllm-ascend/pull/22169/files)在查询时为 open、非 draft、未合入，head 为 `0098214840d1b6d118e602922f17271a9104d13c`，PR 声明的 vLLM 基线为 `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`。PR 描述的上游 G1 指 RFC 的参数保留和布局组，相关改动按本文模型与传输阶段接续验收。

| 此 PR 的内容 | 对应场景 | 后续交付 |
| --- | --- | --- |
| worker 配置上下文、失败收尾和目标重置，磁盘入口增加 NZ 检查 | 阶段 1 会话与支持入口 | 核对 runner 的 LoRA reset 依赖、补 incomplete 上报和完整支持表，覆盖失败后不恢复服务 |
| BF16/FP16 MoE、EPLB/MegaMoe 专家列表、DeepSeek `wo_a`，以及 ND staging + 固定 NZ 运行区 | 阶段 1 MoE，后续模型 NZ 扩展 | 核对 Fused-MC2 实际路径及物理格式、 staging 常驻与峰值；按模型和硬件验收。PR 仍限制 `weight_nz_mode=0` |
| GLM SFA KPool、native MLAPO 和 SFA PROLOG_V3 的派生回填 | 阶段 4 复杂模型 | 进入复杂模型交付，单测与组件证据接续复用；hook 所有权和 Indexer 加载单列 |
| Qwen3.5 HCCL unpacked 加 `layerwise/direct` 参数，参考结果增加 seed 区分 | 阶段 3 Qwen3.5 HCCL unpacked | PR 无 HCCL/IPC 引擎改动。补独立 eager/no-sleep、两条 packed、IPC 与真实失败；packed 仍只收集处于 skip 的 Layerwise 用例 |

PR 报告较早 12 文件版本的 A3/A5 HCCL unpacked 真机结果，其中 A5 明确为 FULL_DECODE_ONLY。CPU UT 结果对应另一候选版本。将其作为首个闭环的已有证据，按最终 head、配对 vLLM SHA、具体模型和运行配置复跑后更新状态。

**现有 Layerwise 权重更新回归台账。** 当前默认模式的用例按配对方式增加 Direct，显式设置两种 reload mode 并保留 Layerwise 原配置。按用例语义判断迁移范围。

| 上游 main 用例 | 当前覆盖 | Direct 跟进 |
| --- | --- | --- |
| [Qwen HCCL](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/e2e/pull_request/two_card/rlhf/weight_transfer/hccl/test_qwen_hccl.py) / [Qwen IPC](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/e2e/pull_request/two_card/rlhf/weight_transfer/ipc/test_qwen_ipc.py) | Qwen3.5-35B-A3B 四层文本 MoE，含 linear/full attention。两后端收集 packed/unpacked，共享 skip 屏蔽 packed | 第三阶段接入 Direct 并对照重跑，先 HCCL unpacked 再扩其余通道。缩层用例用于定位，发布验收登记全层配置 |
| [GLM IPC](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/e2e/pull_request/two_card/rlhf/weight_transfer/ipc/test_glm_ipc.py) | GLM-5.1 缩层缩专家，#19828 增加 GLM-5.3-Flash 缩模和 KDA/SFA/KPool 路径，两者均从 IPC unpacked 起步 | 按型号分别验 Direct，复用 #19828 的模型 fixture 与组件测试，加图 replay、派生值及地址检查 |
| [DeepSeek IPC](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/e2e/pull_request/two_card/rlhf/weight_transfer/ipc/test_deepseek_ipc.py) / [共享 case](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/e2e/common/rlhf/weight_transfer/case.py) | V4 Flash BF16 已定义但标 skip，本版 IPC 入口实际只选 V4.1 Engram host-offload，依赖本地 checkpoint，覆盖 sleep/restore 和非 Engram 更新 | 先恢复 V4 Flash BF16 可执行用例并解决 attention sink 与通信限制。V4.1 offload 待独立支持决策，目前登记 Direct 拒绝或待支持，不以现有用例声称 Engram 值更新 |
| [weight checker](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/e2e/pull_request/two_card/rlhf/state_transitions/test_weight_checker.py) | Qwen3-0.6B，磁盘 reload 恢复和 IPC/HCCL packed/unpacked 校验 | 加 Direct 的小模型入口回归，确认 checker 操作与 checkpoint 恢复语义。checksum 补充 cold B/实际输出与图引用验证 |
| [DeepSeek attention sink UT](https://github.com/vllm-project/vllm-ascend/blob/ed58d21a18922862432ce48574270dcc8438779c/tests/ut/models/test_deepseek_v4_weight_loader_reload.py) | 有明确 Layerwise reload 和 TP loader 测试 | 补 Direct lifecycle、分片与重复更新回归，纳入复杂模型门槛 |
| worker、HCCL、IPC、packed、reference oracle UT | 公共协议、异常路径和辅助组件 | 按受影响语义补 Direct。KV cache 的 layerwise 分配/传输用例属于不同功能，随相关子系统跑回归 |

公共测试先解耦执行模式和 sleep，参考服务与待测服务使用相同模型、布局和执行配置，且从同一份目标权重独立加载。失败路径保持暂停。packed 的共享 skip 按后端核实跨流与生命周期后逐项解除，skip 和未收集均记为覆盖缺口。每个支持的 Direct cell 验冷 B 数值、A→A 与重复更新。图场景额外验原图 replay、地址和物理格式。

---

## 5. 术语表

| 术语 | 含义 |
| --- | --- |
| refit | 在已加载的模型实例上更新权重并完成必要的后处理，使其继续推理。本文的 refit 验证检查热更新后的模型是否与独立冷加载新 checkpoint 一致，图模式还检查原捕获图引用的地址与推理结果 |
| PWAL | `process_weights_after_loading`：转置、量化、打包、融合、算派生 scale 等加载后处理 |
| checkpoint 布局 | loader 要写入的本地 shape/stride/dtype/offset 与加载属性（非完整未分片文件形状） |
| 保址 | 保住图/kernel 真正读取的 storage 与地址；只保参数名或 Python 对象不够 |
| 派生张量 | 由权重算出、被 kernel 直接读、但不是 checkpoint 参数的张量 |
| staging（ND 加载区） | 为强制 NZ 权重额外保留的、供 loader 写入的 ND 缓冲区（候选设计） |
| retired 参数 | 标记退休的源参数：保留注册与数据、可再加载、kernel 参数枚举跳过 |
| NZ / ND | Ascend FRACTAL_NZ 物理排布 / 普通 ND 排布 |
| sleep2 | 本适配目标：按 allocation/tag 保留或丢弃状态；Ascend 具体支持待验证 |

### 5.1 refit 验证如何进行

vLLM 的 `test_reload_refit.py` 按 A→B→A 验证连续更新。先独立冷加载 checkpoint B，保存模型状态和生成结果作为参考。另一个引擎从 A 冷启动，完成预热与图捕获，再经 Direct Reload 加载 B，比较参数、buffer、派生张量和可达的标量状态，以及生成 token/logprobs。随后重新加载 A，核对其结果是否恢复到冷 A。

图模式同时检查原图所引用的张量地址、stride 和 offset，保留旧 storage 引用以识别释放后重新分配的情况，并检查内存占用与更新峰值。本文的“refit fixture”指执行这套流程的测试设施，“冷 B oracle”指独立冷加载 B 得到的参考状态与输出。真实 HCCL/IPC 通信还需对应的 server/trainer 用例提供传输证据。

## 6. vLLM 上游 Direct Reload 的场景与 BF16 验证调研（2026-11-19）

本节核对上游 `main@ecc827ebbf627792e97a44f328fddac95e293129`。本地仅 fetch 后读取远端提交，前文固定源码走读基线仍按各节记录。运行结果引用作者公开的 PR/RFC 报告，本次未执行 GPU/NPU 测试，也未逐个核验历史作业的原始日志。报告中的版本、overlay、硬件与当前 main 分别记录。

### 6.1 BF16 适配了哪些场景

BF16 checkpoint 的未量化推理是本节重点。BF16 输入经 online quant 转成 FP8/FP4、或 BF16 权重搭配 FP8 KV，需额外登记量化后端和 KV dtype。

| 场景 | 当前主线适配 | BF16 验证结论与边界 |
| --- | --- | --- |
| Dense、MoE、MLA、Mamba/attention hybrid | direct begin/load/finish、参数保留、PWAL 重跑和派生回填 | 有 BF16 真实模型 refit 结果，见 6.2。不同模型和 kernel 的消费者需分别核对 |
| 真实新权重 A→B→A | #65347 的 refit 对比独立冷 B，检查模型状态、地址、输出与内存 | BF16 Qwen3-0.6B、DeepSeek-V3 debug、Nemotron hybrid 均有报告。debug checkpoint 的覆盖限于相应结构和配置 |
| NCCL packed、TP2、level-2 sleep | #65799 的完整性检查与 metadata 恢复 | Qwen3 BF16 dense/MoE 有真实 server+trainer 的执行报告，覆盖漏发拒绝、完整重发和 load plan。该报告未明确具体 CUDA graph mode |
| tied/untied embeddings | #66487 finish 保留冷启动时的 tying 关系 | Qwen3-0.6B 有磁盘 reload、不同 lm_head 和 greedy 对照。冷启动已 tied 的对象仍按共享参数语义加载 |
| dummy 初始化与 load plan | #65799 首轮建立加载集合，后续检查完整性，计划写入也上报加载 | 有 Qwen3-30B BF16 NCCL packed 报告。Ascend 自有 receive/plan 路径需单独对接 |
| LoRA 包装后的基座权重更新 | #68345 记录冷启动模块，后续 PWAL 遍历该集合，移除配置级 LoRA 拒绝 | 已报告的实模是 Qwen3-30B W4A16 + LoRA，50 个相关测试通过。BF16 LoRA 全模型结果仍需另找证据或补验 |
| draft/投机推理 | draft lifecycle、派生张量回填，DFlash 测试参数化 direct/layerwise | 当前用例 target Qwen3-4B、draft Qwen3-4B-DFlash-b16，先生成再同权重加载 draft 并检查地址。测试未比较更新后输出，也未走真实传输。按用例存在登记，具体运行需查对应 PR 证据 |
| TP/EP/EPLB/DP | refit fixture 支持多布局 | #65347 两卡五个已报告 case 均为 FP8。BF16 TP2 另有 #65799，RFC 另有 BF16 大模型多节点报告。BF16 EP/EPLB/DP 全矩阵仍待补齐，EPLB refit 未执行 rebalance |
| offload、sharded RDT、torchao、draft_lm_head_quantization | 当前配置入口保留拒绝 | 这些配置需独立专项接入。预处理 weight-cache 冷启动也缺原始 checkpoint layout |

主线新增项需同步到后续 Ascend 基线评估。相对前文 11-17 基线，**#65799 完整性检查、#66487 tying 修复、#67563 构造期 padding 标记、#68345 冷启动模块遍历均已包含**。2.11 的 PR 状态、2.9 的 LoRA 拒绝和“core 无完整性检查”描述仅适用于前文旧 pin。

### 6.2 BF16 模型端到端运行证据

| 模型 | 已报告场景与结果 | 适用范围 |
| --- | --- | --- |
| Qwen/Qwen3-0.6B | #65347 nightly TP1 refit，A→B→A，独立冷 B、逐状态 bitwise、地址/内存、生成输出，早期报告 270 秒通过 | BF16 dense、tied embedding。GPU 图路径，单卡，无 sleep；NCCL/IPC 真实传输另验 |
| inference-optimization/DeepSeek-V3-debug-empty | #65347 nightly TP1 BF16 refit，早期报告 490 秒通过 | BF16 MLA + MoE 的 debug checkpoint，具备完整引擎/生成验证。与生产全量 DeepSeek V3/V4 分别登记 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | #65347 自动单卡 refit，早期报告 213 秒通过 | BF16 Mamba2 + attention hybrid、图路径、A→B→A、冷 B 与内存地址验证 |
| Qwen3-30B-A3B-Instruct-2507 | #65799 GB200，TP2，BF16 trainer，真实 NCCL packed + sleep2。完整更新与冷启动 870/870 参数 digest 一致，漏发被拒绝，完整重发恢复；含 dummy/load plan/保留参数分支 | 真实传输、失败恢复证据。该实验主要重发原 checkpoint，覆盖完整性而非另造任意新 B 的冷 B 等价性 |
| Qwen3-4B-Instruct-2507 | #65799 同环境 dense，TP2 NCCL packed + sleep2，580 参数。漏发 o_proj 或 embedding 被拒绝，完整重发修复 | tied embeddings 和真实遗漏更新，GSM8K/逐参数 digest。图模式未明确，未填报为 FULL/PIECEWISE |
| Qwen3-0.6B（独立 lm_head checkpoint） | #66487 单 L4，nightly cf5cea2 加 PR，B 使 head 与 embed 相等，C 使两者不同，验证 head 仍独立更新，greedy 对齐 Layerwise，新增两测试通过 | 磁盘 reload 特定 tying 回归，PR 未明确 dtype，作为 BF16 模型计划的相关回归，dtype 待确认 |
| Kimi-K3 | RFC 明示 BF16、TP8×2 节点，同权重三次更新，GSM8K 0.951→0.948，表述为张量 diff 0 | RFC 作者历史整栈结果，具体 graph/backend/当前 main 复跑状态未给足。其他 PR 的 Kimi MXFP4 结果另行登记 |
| Nemotron-3.5-Lightning | RFC BF16 hybrid Mamba，GSM8K 0.858→0.857 | 同权重三次的历史报告，型号与 refit 的 Nano-4B 分开 |
| Gemma-4 | RFC BF16，GSM8K 0.948→0.948 | 同权重历史结果，缺少完整传输/graph/并行配置 |
| Ling-3 | RFC BF16 MoE，GSM8K 0.908→0.900 | 同上，分数变化如实记录，避免称 GSM8K bitwise 相等 |
| Inkling | RFC BF16、1.77 TB，GSM8K 0.889→0.892 | 大模型历史结果，仍需绑定具体提交、部署配置和作业 |

在所核对的当前 refit 列表、RFC 表格和相关 PR validation 中，尚未定位 **Qwen3.5 BF16 的上游 Direct 全模型明确运行记录**。Qwen3 系列上游结果可作测试设计参考，Ascend #22169 的 Qwen3.5 证据继续单列。Hy4、Muse 等 dtype 未明确的表格项也未计入 BF16 清单。

### 6.3 仓内测试和实际运行范围

| 测试 | 验证内容 | 已报告的执行证据/限制 |
| --- | --- | --- |
| `test_direct_reload.py` | toy 模型 checkpoint 视图、加载身份、同/新权重、缺参、dummy、padding/offset/共享参数、metadata、失败重开、配置拒绝 | #65347 合并阶段与 load-plan 共 77 通过。#65799 又新增完整性用例。toy 名称中的 MLA/NVFP4 属模拟变换 |
| `test_reload_lifecycle.py`、`test_gpu_worker_weight_transfer.py` | 模式分派、worker 生命周期、失败/目标控制 | #65799 四个 loader 文件报告 173 通过、1 失败。失败是两实验臂共有的 KV-scale case。#68345 相关测试组报告 50 通过，版本不同勿叠加计数 |
| `test_reload_refit.py` | AsyncLLM 实模、A→B→A、冷 B、状态、旧 storage 保活后的地址检查、生成、allocator 与峰值 | #65347 df14510110/job48048 报告自动 5 通过，nightly 单卡 4 通过，两卡 5 通过。后续合并/清理未 GPU 重跑，当前主线同文件存在 |
| `test_load_plan.py`、`test_weight_transfer.py` | 计划写入/manifest、通信协议和 metadata 复用、漏发后恢复 | #65799 NCCL 相关 UT 76 通过。含模拟通信/模型断言，真实传输由其单独 server 实验提供证据 |
| `test_weight_tying.py`、`test_reload.py` | embeddings tying、磁盘 reload、LoRA 等 | #66487 的两个新增测试 2 通过。#68345 W4A16+LoRA 相关组 50 通过 |
| `test_draft_weight_reload.py` | DFlash draft 同权重加载后检查地址 | 代码参数化两种模式，按测试设计覆盖登记。本节未取得该具体 case 的独立作业通过记录 |
| `test_reload_moe_kernels.py`、量化/参数身份相关测试 | 部分 MoE kernel runtime 地址、量化布局和对象身份 | 为 P2 提供参考，其量化格式与 BF16 结果分别登记 |

`test_reload_refit.py::_stream_reload` 直接在 worker 上调用 start、按 16 个 device tensors 一批执行 `model.load_weights`、再 finish。配置 `backend=ipc` 使生命周期有关分支生效，实际数据通过本地文件读取搬到 GPU，未进行 IPC handle 交换或 NCCL broadcast。这组测试可作为 Direct 模型正确性 oracle，通信后端仍需真实 server/trainer 验证。

CI 当前有 B200 自动 refit、单卡 nightly extra 和两卡 nightly 三步。BF16 Nemotron 在自动步骤，BF16 Qwen3/DeepSeek debug 在 nightly extra。两卡五个 refit case 为 FP8，sleep2 refit 为 gpt-oss MXFP4。当前 case 未参数化 eager/各图模式，也没有 BF16 × packed/unpacked × NCCL/IPC 的八格全覆盖。nightly optional 配置表示执行入口可用，实际是否成功以对应 job 为准。
