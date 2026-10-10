# 2026 年 10 月 1 日至 7 日 tool calling 与 vLLM Router 变更阅读记录

# Tool Calling

## 新特性与能力扩展

### （关注可覆盖的模型范围以及存量模型是否需要调整启动命令）基于 checkpoint response_template 的 hf parser

新增 `hf` parser，根据 checkpoint 元数据解析正文、思考内容和工具调用，减少对模型专属 parser 的依赖。其适配依据是 checkpoint 的输出协议元数据；模型的 `tokenizer_config.json` 必须包含 vLLM 支持的 `response_template`。

Python 前端可使用 `--enable-auto-tool-choice --tool-call-parser hf --reasoning-parser hf`。工具解析和思考解析可单独启用，但 `hf` 不能与其他 tool/reasoning parser 混用。

Rust 前端通过 `--tool-call-parser hf --reasoning-parser hf` 显式选择该解析器。

#### 解决的问题：由 checkpoint 声明输出协议

不同模型可能以 JSON、XML 或带特殊分隔符的文本输出工具调用，思考内容与最终回答的边界也各不相同。此前通常需要选择匹配模型的专属 tool/reasoning parser。`hf` 从 checkpoint 的 `response_template` 读取区域边界和工具调用转换规则，交由通用执行器解析。模型仍需生成符合模板的输出；仅指定 `hf` 不会使模型获得原本没有的工具调用能力。[#58604](https://github.com/vllm-project/vllm/pull/58604)、[#59005](https://github.com/vllm-project/vllm/pull/59005)

三类配置各有职责：`chat_template` 将消息和工具定义渲染为输入 prompt；`response_template` 描述生成结果的解析规则；请求 tools 中的 `parameters` schema 为参数类型转换提供依据。输入模板与输出解析规则需要匹配模型协议，参数类型转换也不等于生成时的严格 schema 约束。`--tool-call-parser hf`、`--reasoning-parser hf` 和 `--tokenizer-mode hf` 分别控制工具解析、思考解析和 tokenizer 模式。

模板主要描述以下信息，具体字段以 checkpoint 的真实元数据为准：

| 信息 | 作用 |
| --- | --- |
| `start_anchor` / `start_anchor_pattern` | 在 prompt 中定位当前 assistant 轮，避免将历史工具调用识别为本轮输出 |
| `fields` | 定义 `content`、`thinking`、`tool_calls` 等区域 |
| `open` / `open_pattern`、`close` / `close_pattern` | 定义区域起止分隔符；正则表达式可通过命名捕获组提取函数名等信息 |
| `content` / `content_args` | 按 `text`、`json`、`xml-inline`、`kv-lines` 等规则解析区域内容 |
| `transform` | 将捕获值和解析结果映射到输出对象，如 `function.name` 和 `function.arguments` |
| `repeats` / `join` | 定义重复区域及其聚合方式；Python 与 Rust 前端的支持范围不同 |

### 兼容性修复

| 日期 | 类型 | 变更与边界 | 来源 |
| --- | --- | --- | --- |
| 10-06 | Rust/Python 行为对齐 | Rust 前端向聊天模板透传工具的 `defer_loading` 字段，与 Python 前端保持一致 | [#60203](https://github.com/vllm-project/vllm/pull/60203) |

### 模型 parser 与 grammar 重构

| 日期 | 类型 | 变更与用户可见行为 | 来源 |
| --- | --- | --- | --- |
| 10-01 | Step parser 迁移 | Step-3.5/3.7 通过继承 `Qwen3Parser` 接入流式解析引擎 | [#59321](https://github.com/vllm-project/vllm/pull/59321) |
| 10-06 | MiniMax M3 parser 迁移 | 从 PyO3 bridge 迁至声明式解析引擎，正常输出保持一致；畸形或截断调用保留出错前已解析的参数，替代旧 bridge 回退原文或丢弃后续流的行为 | [#59743](https://github.com/vllm-project/vllm/pull/59743) |
| 10-06 | 旧 bridge 移除 | M3 作为最后一个生产使用方完成迁移后，删除 `vllm-tool-parser-py`、`RustToolParser` adapter、`_rust_tool_parser` 注册及相关依赖和 CI 配置 | [#59744](https://github.com/vllm-project/vllm/pull/59744) |

## 问题修复

### Kimi、GLM 解析与 parser 兼容性修复

| 日期 | 问题 → 修复后行为 | 来源 |
| --- | --- | --- |
| 10-01 | 未完成的 UTF-8 字节后，独立 token 的来源位置被错误标在旧文本段开头 → 对可确认独立的后续 token，改为标在自身首字节；解码文本不变，为 K3 基于 token 身份的解析提供基础 | [#58357](https://github.com/vllm-project/vllm/pull/58357) |
| 10-01 | K3 思考内容中由普通 BPE token 拼出的标记被误认为通道边界 → channel/call 边界必须匹配专用 token ID，相同字面文本保留为当前通道内容 | [#58358](https://github.com/vllm-project/vllm/pull/58358) |
| 10-02 | Rust GLM 字符串参数被裁去首尾空白，导致缩进、换行或编辑目标改变 → `glm45` / `glm47` 在完整和流式解析中保留首尾空白，数值仍按 schema 转换 | [#59654](https://github.com/vllm-project/vllm/pull/59654) |
| 10-07 | parser 与 tokenizer 不兼容时，服务仍可启动，首次 chat 请求才返回 500 → 启动时校验组合 parser，以包含选项和模型信息的 `TypeError` 提前报错；覆盖 reasoning/Harmony，`tokenizer=None` 时跳过 | [#59749](https://github.com/vllm-project/vllm/pull/59749) |

K3 不提供纯文本回退：缺少 token 来源信息时，标记被视为正文。内部 argument/json 块及输出 grammar 仍按文本解释，字符串中的闭合标记歧义尚未完全解决。[#58358](https://github.com/vllm-project/vllm/pull/58358)

## vLLM Ascend 相关变更

本期未列入直接相关的 toolcall 新特性或修复。

# vLLM Router

本期 main 分支无新特性或问题修复。
