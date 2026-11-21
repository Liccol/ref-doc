# LoRA 源码索引

所有链接固定到配套 vLLM commit `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，不指向浮动 main。行号由该提交的本地源码逐项核对。

建议结合 [主链路说明](README.md) 阅读；Ascend 的固定版本源码见 [Ascend 接入说明](ascend-and-minimax.md)。

| 阶段 | 源码入口 | 阅读重点 |
| --- | --- | --- |
| 配置入口 | [`EngineArgs.create_engine_config`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/engine/arg_utils.py#L2748) | enable_lora 决定是否构造 LoRAConfig |
| 配置定义 | [`LoRAConfig`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/config/lora.py#L30) | 默认值、容量校验、模型/kernel 校验及 dtype 解析 |
| 组合校验 | [`VllmConfig`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/config/vllm.py#L1551) | 调用 LoRA 与模型/kernel 的兼容检查 |
| 模型协议 | [`SupportsLoRA / supports_lora`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/model_executor/models/interfaces.py#L719) | 模型声明及能力检查 |
| 模型能力 | [`ModelConfig.is_lora_supported`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/config/model.py#L2046) | 读取模型注册能力 |
| 静态注册 | [`init_app_state 中的调用点`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/entrypoints/launchers/api_server/app_state.py#L86) | 服务启动时加载静态 adapter |
| 服务加载 | [`OpenAIServingModels.load_lora_adapter`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/entrypoints/openai/models/serving.py#L167) | 分配逻辑 ID，并调用 engine_client.add_lora |
| 动态接口 | [`LoRA API router`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/entrypoints/serve/lora/api_router.py#L30) | 动态管理接口的开关 |
| 异步引擎 | [`AsyncLLM.add_lora`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/engine/async_llm.py#L1261) | 转到 CoreClient |
| 引擎核心 | [`EngineCore.add_lora`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/engine/core.py#L1243) | 转到 executor |
| 跨 worker | [`Executor.add_lora`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/executor/abstract.py#L407) | collective_rpc 广播管理操作 |
| Runner 加载 | [`GPUModelRunner.load_model`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/worker/gpu/model_runner.py#L405) | 基座装载后加载 LoRA 执行结构 |
| Runner mixin | [`LoRAModelRunnerMixin`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/worker/lora_model_runner_mixin.py#L28) | worker manager 创建、batch 激活、dummy warmup |
| Worker 管理 | [`WorkerLoRAManager / LRUCacheWorkerLoRAManager`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/worker_manager.py#L27) | 读取 checkpoint、管理缓存和激活 |
| Model 管理 | [`LoRAModelManager`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/model_manager.py#L76) | 注册缓存、设备槽位、Punica 与层包装 |
| 层替换 | [`_create_lora_modules`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/model_manager.py#L476) | 遍历、过滤、处理别名并替换子模块 |
| 包装选择 | [`from_layer`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/utils.py#L116) | can_replace_layer 和 create_lora_weights |
| 设备实现 | [`get_punica_wrapper`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/punica_wrapper/punica_selector.py#L13) | 平台返回类路径，再动态实例化 |
| 元数据 | [`PunicaWrapperBase`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/punica_wrapper/punica_base.py#L120) | token/prompt 索引与 prefill 元数据 |
| ID 转换 | [`convert_mapping`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/punica_wrapper/utils.py#L52) | 逻辑 adapter ID 转设备槽位 |
| Adapter 配置 | [`PEFTHelper`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/peft_helper.py#L21) | 训练配置、rank/alpha pattern、合法性检查 |
| Adapter tensor | [`LoRAModel.from_local_checkpoint`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/lora_model.py#L197) | 读取 safetensors/bin/pt，名称映射与 dtype 转换 |
| Scaling | [`LoRALayerWeights.optimize`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/lora_weights.py#L37) | 将 scaling 合并到 B |
| 设备激活 | [`activate_adapter`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/model_manager.py#L355) | set_lora / reset_lora，写入槽位 |
| 调度 | [`Scheduler 的 LoRA 调度段`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/core/sched/scheduler.py#L1014) | 限制 batch 中不同 adapter 的数量 |
| Batch 映射 | [`LoraState`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/v1/worker/gpu/lora_utils.py#L75) | 按 token/logits 数展开 adapter ID |
| Linear 计算 | [`BaseLinearLayerWithLoRA`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/layers/base_linear.py#L70) | 基座输出叠加 Punica 低秩结果 |
| Column/QKV | [`Column / Merged / QKV wrappers`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/layers/column_parallel_linear.py#L466) | 输出切片、Q/KV shard 与复制 |
| Row | [`RowParallelLinearWithLoRA`](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/layers/row_parallel_linear.py#L22) | A 输入分片与输出归约 |
