# 2026 年 10 月 1 日至 7 日 tool calling 与 vLLM Router 变更阅读记录

整理日期：2026-10-10。研究仓库：[vllm](https://github.com/vllm-project/vllm)、[vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[router](https://github.com/vllm-project/router)。本次重新核对三个 main 分支的区间提交、相关 PR 正文及最终 diff，按九月大纲整合原正文与补充项。

## 1. 时间与统计口径

- 沿用九月 UTC 口径：`2026-10-01 00:00:00 ≤ merged_at < 2026-10-08 00:00:00`。表格日期是 PR 合并日期，不是创建日期或版本发布日期。
- vLLM / Ascend 聚焦 toolcall 新子特性、问题修复，以及直接影响工具解析、参数约束、reasoning 交接和工具往返的共享链路；Router 覆盖全部新特性与问题修复。模型内部 MoE 路由不纳入服务 Router。
- 分页读取 `sha=main` 区间 commits：vLLM 349 条、Ascend 39 条、Router 0 条；交叉检索区间已合并 PR：分别为 350、39、0 项，均 `incomplete_results=false`。提交与 PR 不是同一统计量；vLLM 搜索额外返回的 [#59651](https://github.com/vllm-project/vllm/pull/59651) 是 HiSparse 相邻工作，不属于 toolcall 范围。
- 本次 main 快照：[vLLM 3c802af3](https://github.com/vllm-project/vllm/commit/3c802af3cf6263d0ef6fb356a7884680ecf15f04)、[Ascend 72d48aef](https://github.com/vllm-project/vllm-ascend/commit/72d48aef09ef692c8423baf823d6a171f4062bed)、[Router 91b65ee4](https://github.com/vllm-project/router/commit/91b65ee4bdfbe98863a32a2205dced5dbb249ffe)。快照固定检索上下文，正文只统计指定区间。
- 筛查完整提交及 PR 标题，阅读主题相关正文与最终 diff；#58358、#59744 等堆叠 PR 的 base 虽不是 main，对应提交已出现在 main 区间历史中。不是全部源码逐行审计。
- 正文与最终代码冲突时按 diff 记录。**主线合并不等于 release 或 Ascend 镜像已经包含。** 本次未运行 parser 测试、GPU/NPU serving 或性能基准，作者验证记录单独归因。

复查入口：[vLLM 已合并 PR](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)、[Ascend 已合并 PR](https://github.com/vllm-project/vllm-ascend/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)、[Router 已合并 PR](https://github.com/vllm-project/router/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)。

## 2. 主要结论

| 主题 | 本期变化 | 阅读或升级时的关注点 |
| --- | --- | --- |
| 模板驱动 toolcall | Python、Rust 新增 hf response-template parser | 两端流式发出时机与严格约束边界不同 |
| 工具调用约束 | GLM 非 strict 工具新增浅层 tag；prompt 与 grammar 工具集合对齐 | 浅层约束不等于完整 schema，关闭总开关仍禁用 tag |
| 模型解析 | Step、MiniMax M3 迁移 engine；K3、GLM、Mistral 修复 | 关注 token 身份、参数空白、异常 JSON 和残缺调用 |
| schema 正确性 | Rust 共享 SchemaRoot；非法 $defs、深嵌套和共享 grammar 修复 | Python/Rust、Guidance/XGrammar 的边界分别判断 |
| API 与拆分前端 | Harmony call ID 一致性，render/generate 状态和 derender 上下文修复 | 客户端事件名及工具解析上下文需要核对 |
| Ascend | GLM-5.3 always-on reasoning 回移 | 影响共享 GLM parser；toolcall 集成未做端到端验证 |
| Router | 区间 main 无提交、已合并 PR 为 0 | 没有本期新特性或 bugfix，vLLM 变化不自动等于 Router 集成 |

## 3. vLLM：新特性与能力扩展

### 3.1 checkpoint response_template 驱动的 hf parser

新增依据 checkpoint 元数据解析 content、reasoning 和 tool calls 的能力，减少模型专属 parser 的需要。Python 与 Rust 分别实现。

| 日期 | 层次 | 变更与边界 | 来源 |
| --- | --- | --- | --- |
| 10-01 | chat_parsing 基础事件 | region_open/close 带 delimiter span，open 带 captures；新增 malformed-region 事件，流式继续解析后续内容，非流式仍在首个异常抛错 | [#58603](https://github.com/vllm-project/vllm/pull/58603) |
| 10-01 | Rust hf unified parser | 原生执行 tokenizer_config.json 的 response_template；显式选 hf，auto 不自动选，部分 Python regex/template 字段不支持，边界 pattern 需要有限 literal prefixes | [#59005](https://github.com/vllm-project/vllm/pull/59005) |
| 10-02 | Python hf parser | 从运行时 tokenizer 元数据解析工具与 thinking；Chat、Responses、derender 传入渲染后 prompt，缺失/无效 metadata 校验失败 | [#58604](https://github.com/vllm-project/vllm/pull/58604) |

Python 可用 `--enable-auto-tool-choice --tool-call-parser hf --reasoning-parser hf`。tool/reasoning 可分别选择，但不能与其他 parser 混搭，Harmony 保持既有 parser。每项调用在 region 关闭并成功解析后整项发出；流结束时未关闭 region 尝试解析，失败警告并丢弃，未知函数名可透传。**strict tools、required/named tool choice、parallel_tool_calls=false 暂被拒绝**，因缺兼容 grammar。[#58604](https://github.com/vllm-project/vllm/pull/58604)

Rust 显式选 `--tool-call-parser hf --reasoning-parser hf`；文本/reasoning 增量发出，opener 捕获函数名时可先发调用开始，参数在 region 关闭后发出。尚无增量参数、token-aware 模板边界及模板派生 structural grammar；strict/required 缺少该格式的生成约束，**不能套用 Python 的“拒绝”结论**。[#59005](https://github.com/vllm-project/vllm/pull/59005)

### 3.2 GLM-4.7 非 strict 工具的浅层约束

**10-02：[PR #56403](https://github.com/vllm-project/vllm/pull/56403)** 将 GLM builder 纳入本地 registry。strict 工具保持参数 schema 约束，非 strict 工具限制调用标签、函数名、简单 key 和基础 value 类型；允许参数省略、重复及任意顺序，复杂 schema 回退到更宽的原生内容，全 strict 请求仍委托 XGrammar builtin。

最终代码将 `Glm47MoeModelToolParser.default_tool_strict_level` 设为 FUNCTION，operator 保持 AUTO 时采用此默认值，普通 auto 调用也可获得浅层 tag；PARAMETER 提高全部工具参数约束。**VLLM_ENFORCE_STRICT_TOOL_CALLING=0 仍禁用 structural tags。** PR 正文“关闭开关时安装 grammar”是早期描述，以最终默认级别和 flag-off 测试为准。[#56403 最终变更](https://github.com/vllm-project/vllm/pull/56403/files)

浅层 tag 不保证必需参数全部出现，不等于完整 JSON Schema 校验。后续工具集合对齐见第 4.2 节。

### 3.3 动态工具与延迟加载兼容

| 日期 | 类型 | 变更与边界 | 来源 |
| --- | --- | --- | --- |
| 10-02 | Anthropic 扩展/修复 | /v1/messages 接受 system 中 tool_addition/tool_removal；加入工具解除 defer_loading，移除工具隐藏，tool_definition 可声明工具；仅 system role，未声明引用返回 400，MCP reference 被拒绝，不新增 connector | [#57693](https://github.com/vllm-project/vllm/pull/57693) |
| 10-06 | Rust/Python 对齐 | Chat DTO 透传工具层和 function 层 defer_loading，function 层优先，未设置时不输出字段；仅模板扩展，不新增服务端 tool search，parser/grammar/choice 不因此禁用 deferred 工具 | [#60203](https://github.com/vllm-project/vllm/pull/60203) |

### 3.4 模型 parser 与 grammar 重构

| 日期 | 类型 | 变更与用户可见行为 | 来源 |
| --- | --- | --- | --- |
| 10-01 | Step 迁移兼修复 | Step-3.5/3.7 以 Qwen3Parser 小子类接入 streaming engine，改善 reasoning 尾空白、当前轮边界及参数转换 | [#59321](https://github.com/vllm-project/vllm/pull/59321) |
| 10-01 | Rust 参数 grammar 重构兼修复 | 拆分通用 schema resolution 和模型 ArgumentSyntax；K3 按声明顺序生成，required 参数必须出现，先解析 $ref 再选 type；含闭合标记的 string enum 只丢对应分支 | [#59395](https://github.com/vllm-project/vllm/pull/59395) |
| 10-06 | MiniMax M3 迁移 | PyO3 bridge 迁至声明式 engine，正常输出保持一致；畸形/截断调用保留出错前参数，改变旧 bridge 原文回退或丢后续流行为 | [#59743](https://github.com/vllm-project/vllm/pull/59743) |
| 10-06 | 旧 bridge 移除 | 删除 vllm-tool-parser-py、RustToolParser adapter、_rust_tool_parser 注册及依赖/CI，最后生产用户 M3 已迁移；独立 Rust frontend/parser 保留 | [#59744](https://github.com/vllm-project/vllm/pull/59744) |

Step required/named 原生 XML 由九月 [#51810](https://github.com/vllm-project/vllm/pull/51810) 修复，本期不重复归因。M3 在 invoke 关闭后整项发参数，不将迁移写成增量参数新能力；未来 _rust_* 构建 plumbing 仍保留。

## 4. vLLM：bug 修复与行为变化

### 4.1 Kimi、GLM、Mistral 与启动校验

| 日期 | 问题 → 修复后行为 | 来源 |
| --- | --- | --- |
| 10-01 | 待完成 UTF-8 后独立 token 锚到旧段开头 → 可证明独立的后续 token 锚到自身首字节，解码文本不变，是 K3 token-aware 基础 | [#58357](https://github.com/vllm-project/vllm/pull/58357) |
| 10-01 | K3 reasoning 中普通 BPE 拼出 marker 被误关闭 channel → channel/call 边界要求专用 token ID，普通 lookalike 保持正文 | [#58358](https://github.com/vllm-project/vllm/pull/58358) |
| 10-02 | Rust GLM string 参数 trim 改变缩进、换行和编辑目标 → glm45/glm47 保留首尾空白，完整/流式生效，数值仍按 schema 转换 | [#59654](https://github.com/vllm-project/vllm/pull/59654) |
| 10-06 | Mistral pre-v11 异常 JSON 使请求失败，string arguments 重复编码 → 裸 object 作单调用，缺失/非 string name 为空字符串，string arguments 原样保留；无调用 JSON 回到 content，第二 marker 后作 trailing text，流式/非流式对齐 | [#54844](https://github.com/vllm-project/vllm/pull/54844) |
| 10-07 | parser/tokenizer 不兼容但服务启动健康，首次 chat 才 500 → 启动时实例化组合 parser，提前以带 flag/model 的 TypeError 失败；覆盖 reasoning/Harmony，tokenizer=None 时跳过 | [#59749](https://github.com/vllm-project/vllm/pull/59749) |

K3 无 text-only fallback：没有 token attribution 时 marker 被视作正文。内部 argument/json 块及 output grammar 仍按文本解释，闭合标记歧义未全部解决。[#58358](https://github.com/vllm-project/vllm/pull/58358)

### 4.2 schema、tool choice 与共享约束正确性

| 日期 | 问题 → 修复后行为 | 来源 |
| --- | --- | --- |
| 10-01 | speculative draft 缺失后 padding grammar 行是 full mask，worker 可接受未约束 draft → 记录实际 constrained leading draft 数，超出位置失效并重采样；handoff 身份根因另行处理 | [#54442](https://github.com/vllm-project/vllm/pull/54442) |
| 10-05 | 非 Harmony prompt 与 grammar 工具集合不一致，builtin-only 空 tag、namespace 函数无规则 → get_function_tools 共用于 prompt/tag/旧 required schema；namespace__name 展平，无函数时 auto 不加 grammar，required/named 返回 tool_choice 400，Harmony 保留 builtin | [#59879](https://github.com/vllm-project/vllm/pull/59879) |
| 10-05 | Guidance 禁额外属性遍历改写 const/enum/default/examples 字面值及 properties 映射 → 只递归 schema-bearing keywords，保持字面值和属性映射 | [#58709](https://github.com/vllm-project/vllm/pull/58709) |
| 10-06 | 工具 $defs=null/list/string 或同名不同定义产生 500 → VLLMValidationError 返回 tools 参数 400，正常 definitions 提升保持 | [#54850](https://github.com/vllm-project/vllm/pull/54850) |
| 10-06 | 深 schema/tag 递归 500或同步转换阻塞 API → 转换前迭代检查深度并返回 400；普通 JSON 64 层，structural tag 额外允许 16 层包装 | [#60036](https://github.com/vllm-project/vllm/pull/60036) |
| 10-06 | XGrammar 多分支 allOf 实际约束失效 → 递归识别长度≥2 的 allOf，显式校验报告不支持；不新增 allOf 支持 | [#59061](https://github.com/vllm-project/vllm/pull/59061) |
| 10-07 | Rust 参数转换忽略 $ref，nested object 返回 JSON string → grammar/coercion 共用 SchemaRoot，统一本地 ref、nullable、enum/const、单 schema allOf 和类型别名，按实际值递归转换 | [#59408](https://github.com/vllm-project/vllm/pull/59408) |

共享约束修复与 schema-constrained toolcall 相关，但 Guidance 配置、XGrammar 校验和 Rust SchemaRoot 不能互相套用支持结论。Rust 不可解析/循环/远程引用宽松回退、不联网获取；DSML/K3 仍按 string/type 标签解码。

深度计算 JSON object/array 容器，不等于业务属性层数。以 [#60036 最终 diff](https://github.com/vllm-project/vllm/pull/60036/files) 的 MAX_STRUCTURED_OUTPUT_JSON_NESTING=64、STRUCTURAL_TAG_WRAPPER_NESTING=16 为准。

### 4.3 流式、marker 与 reasoning 状态

| 日期 | 问题 → 修复后行为 | 来源 |
| --- | --- | --- |
| 10-07 | 完整 marker 被 safe-text 当成等待输入，Qwen lookalike 缓冲到流结束 → 完整 marker 返回 Backtrack，部分/空输入仍 Incomplete；Qwen 停止条件带必要换行，非合法 opener 及时作为正文输出，hf event loop 适配 | [#59563](https://github.com/vllm-project/vllm/pull/59563) |
| 10-07 | engine-based reasoning 且无 tool parser 时，reasoning 结束 flush 文本无人发出 → 作为 content 发出，有 tool parser 仍交工具阶段；修复多 token delta 的流式/非流式差异 | [#58911](https://github.com/vllm-project/vllm/pull/58911) |

#58911 的触发条件是没有 tool parser，作为共享交接修复记录，不称为工具提取新能力。

### 4.4 Responses 与 Anthropic 协议

| 日期 | 问题 → 修复后行为 | 来源 |
| --- | --- | --- |
| 10-01 | response.reasoning_part.added/done 不符合官方 SDK schema → response.content_part.added/done，part.type=reasoning_text，覆盖 Simple/Harmony | [#59652](https://github.com/vllm-project/vllm/pull/59652) |
| 10-02 | Anthropic 动态工具块被拒绝/deferred 工具仍隐藏 → 解析 system 工具变更，详细边界见 3.3 | [#57693](https://github.com/vllm-project/vllm/pull/57693) |
| 10-04 | Harmony completed 重建产生新 item id/call_id，客户端无法对应调用 → 按完成消息对象身份匹配记录、复用 ID，reasoning 使用 rs_ 前缀 | [#59859](https://github.com/vllm-project/vllm/pull/59859) |

旧 reasoning_part 事件监听需要迁移，正文讨论的过渡开关未实现。Harmony 内容和 incomplete 状态仍由重建得出，未流出的 zero-delta 项可新建 ID；消息级匹配避免 builtin python 重建成 reasoning 后误取下一消息 ID。九月 [#59307](https://github.com/vllm-project/vllm/pull/59307) 修非 Harmony，本期补 Harmony。

### 4.5 拆分式前端：render / generate / derender

这些影响工具解析上下文和状态，属于 vLLM 前端，不直接算 Router 集成。

| 日期 | 问题 → 修复后行为 | 来源 |
| --- | --- | --- |
| 10-02 | derender 无 prompt context，SentencePiece 首 token 丢空格，batch 截断 UTF-8 补乱码 → prompt tail 初始化 decode window；从请求或 generate response 取 prompt_token_ids，均缺失则旧行为并警告，统一增量解码/特殊 token 选项 | [#59046](https://github.com/vllm-project/vllm/pull/59046) |
| 10-05 | render→generate 丢 reasoning_ended/reasoning_parser_kwargs，约束起点与 thinking 配置不同 → 透传两字段，保持 engine/frontend 一致 | [#60062](https://github.com/vllm-project/vllm/pull/60062) |
| 10-06 | GPU-less render 忽略默认 reasoning parser，gpt-oss 泄漏 Harmony markup → 共用 structured_outputs config 构建，使用解析后配置，显式 flag 优先 | [#54835](https://github.com/vllm-project/vllm/pull/54835) |

#54835 正文“等待 streaming derender #50550”是历史上下文；[#50550](https://github.com/vllm-project/vllm/pull/50550) 九月已合并，不把历史 501 重新写作本期限制，也不额外宣称所有模型组合已经验证。

## 5. vLLM Ascend：直接相关变更

| 日期 | 类型/分支 | 变更与意义 | 来源 |
| --- | --- | --- | --- |
| 10-05 | main，GLM parser 回移 | 回移上游 #56994 always-on reasoning 模板检测和 constructor 修正；thinking=false/enable_thinking=false 不再错误关闭 extraction、把 scratchpad 放入 content | [#17876](https://github.com/vllm-project/vllm-ascend/pull/17876) |

patch 包装共享 Glm47MoeParser constructor，涉及 glm45/glm47 reasoning adapters 与 glm47 tool adapter 的解析状态。匹配 always-on 模板后在 kwargs 副本归一化开关，保留其余配置及调用约定；已有 helper/已包装时跳过，旧模板保留行为。[最终 diff](https://github.com/vllm-project/vllm-ascend/pull/17876/files)

**不提供关闭模型思考的能力。** 作者报告实际目标 tokenizer 离线 parser A/B：12/12 泄漏用例修复、2/2 正常控制通过，未加载模型权重。NPU 推理、HTTP serving、tool calling、结构化输出集成及部署中自动加载 patch 尚未验证，不将离线结果写成 toolcall 端到端验证。[#17876](https://github.com/vllm-project/vllm-ascend/pull/17876)

上游 [#56994](https://github.com/vllm-project/vllm/pull/56994) 是回移来源，不计作本期 vLLM 新增。Ascend 其余区间提交主要为 KV cache、PCP/DCP、算子、模型执行及部署文档，未找到其他直接 toolcall 项，不混入本表。

## 6. vLLM Router：新特性

**本区间无 main 新特性落地项。** 已合并 PR 为 0，main 区间 commits 返回空数组。九月 WASM、Program Progress-TTL、gRPC Engine-Client、NIXL push 和 tokenizer L0 cache 不重复计数。

来源：[已合并 PR 检索](https://github.com/vllm-project/router/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)、[main 区间 commits API](https://api.github.com/repos/vllm-project/router/commits?sha=main&since=2026-10-01T00%3A00%3A00Z&until=2026-10-07T23%3A59%3A59Z&per_page=100)。

## 7. Router：bug 修复

**本区间无 main bugfix 落地项。** 相同区间提交/PR 检索没有可补充的直接提交。九月 [#283](https://github.com/vllm-project/router/pull/283) 的 gRPC adapter 主动 tool calling 限制，没有本期代码证据表明解除；vLLM 新增 parser 或删除 bridge 不自动改变 adapter。

## 8. Router：其余区间已合并项

本区间无 main 重构、性能、CI、文档或发布流程提交可记录。开放 PR、其他分支 push、repository updated_at 不替代 main 落地证据。当前 main 快照与区间提交分别判断，不用当前最新提交推断指定周变化。

## 9. 建议的代码阅读顺序

1. **模板驱动 toolcall**：#58603 events → #58604 Python adapter → #59005 Rust hf；比较 prompt anchor、发出时机和约束边界。
2. **工具调用约束**：#56403 默认严格级别/GLM builder → #59879 get_function_tools；区分外壳、浅层参数与完整 schema。
3. **schema 与参数转换**：#59395 ArgumentSyntax → #59408 SchemaRoot → #54850/#60036/#59061 校验边界。
4. **模型解析**：#58357 attribution → #58358 K3 guards → #59563 safe-text；再看 #59654 GLM 空白和 #54844 Mistral JSON。
5. **迁移与 bridge**：#59321 Step → #59743 M3 converter → #59744 删除；区分已有 forced tool 支持与错误路径变化。
6. **Agent 协议**：#57693 动态工具 → #60203 defer_loading → #59859 call ID，客户端核对 #59652 事件迁移。
7. **拆分前端与 Ascend**：#54835 默认 parser → #60062 状态 → #59046 decode context → Ascend #17876 shared constructor patch。

以上为阅读建议。源码从 PR Files changed / merge commit 查看，避免以持续变化的 main 代替合并时实现。

## 10. 未计入已落地能力的内容

- **重构/维护不作新能力**：[#59393](https://github.com/vllm-project/vllm/pull/59393) snapshot outline、[#59762](https://github.com/vllm-project/vllm/pull/59762) 旧 Transformers 清理、[#60073](https://github.com/vllm-project/vllm/pull/60073) derender parse/assemble 抽取、[#57780](https://github.com/vllm-project/vllm/pull/57780) SDK 测试兼容。
- **通用前端相邻项不展开**：[#58588](https://github.com/vllm-project/vllm/pull/58588) text output mode、[#58181](https://github.com/vllm-project/vllm/pull/58181) generate logprobs 协议、[#47933](https://github.com/vllm-project/vllm/pull/47933) abort、[#59072](https://github.com/vllm-project/vllm/pull/59072) Cohere logprobs、[#60309](https://github.com/vllm-project/vllm/pull/60309) Cohere 4xx、[#59419](https://github.com/vllm-project/vllm/pull/59419) billing prompt 清理，不直接写成工具调用能力。
- **vLLM gRPC 不作 Router 变更**：[#59659](https://github.com/vllm-project/vllm/pull/59659) Rust gRPC port、[#59837](https://github.com/vllm-project/vllm/pull/59837) forbidden tokens/cache usage，不证明 Router 主动 toolcall 已支持。
- RFC、未合并 PR、依赖来源不算本期实现；正文保留 draft/待验证表述时，合并状态与测试完成度分别记录。
- 未推断首次包含各 PR 的 release tag；升级需核对 merge commit、发布分支及镜像依赖。本次没有运行模型质量/性能复现。
