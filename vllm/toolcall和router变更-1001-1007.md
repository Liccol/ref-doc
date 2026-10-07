# 2026 年 10 月 1 日至 7 日 tool calling 与 vLLM Router 变更阅读记录

整理日期：2026-10-07。研究仓库：[vllm](https://github.com/vllm-project/vllm)、[vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[router](https://github.com/vllm-project/router)。格式沿用九月记录。

> **本次为截至抓取时刻的暂定周报。** 抓取于 2026-10-07 14:59–15:03 UTC（北京时间 22:59–23:03）；UTC 的 10 月 7 日尚未结束，不能把本次结果视为全天最终统计。2026-10-08 晚间定时运行应重新核对本区间、补齐晚到变更，并更新同一文件。

## 1. 时间与统计口径

- 区间沿用九月的 UTC 口径：`2026-10-01 00:00:00 ≤ merged_at < 2026-10-08 00:00:00`。下列表格日期是 UTC 合并日期，不是 PR 创建日期或版本发布日期；本次实际可见范围截止上述抓取时刻。
- 交叉检索 tool / parser / reasoning / function / MCP / DSML / structural tag / grammar / Responses / derender / chat_parsing / tool_choice / tool-calling 标签等；阅读 PR 正文，并通过 GitHub REST 的 merged_at 核对时间。
- 对 GLM 浅层 grammar、模板驱动 hf parser、模板与 grammar 工具集合对齐、PyO3 bridge 移除、render→generate 状态、Kimi K3 token-aware parsing、schema 深度限制及 Ascend 回移进一步阅读关键 diff。
- 本文归纳 **vLLM 26 项主题及相邻变更、Ascend 1 项直接相关回移**。Ascend 当期全部 39 个已合并 PR 的标题也进行了筛查。vLLM 是主题检索，不是其全部提交的逐行审计，标题/正文未关联关键词的变更仍可能遗漏。
- Router 对整个区间检索所有已合并 PR，并核对默认分支 main 的 commits API：截至抓取时刻，**已合并 PR 为 0、区间 main 提交为 0**。
- 主体以已合并代码为依据；RFC、未合并 PR、依赖来源不计为本期落地项。main 合并不等于某个 release / Ascend 镜像已经包含。性能和测试是作者记录，本次未执行 GPU/NPU serving 或模型质量复现。

复查入口：[vLLM 区间已合并 PR](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)、[Ascend 区间已合并 PR](https://github.com/vllm-project/vllm-ascend/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)、[Router 区间已合并 PR](https://github.com/vllm-project/router/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07)。

## 2. 主要结论

| 主题 | 本期变化 | 阅读/使用时的关注点 |
| --- | --- | --- |
| 模板驱动解析 | Python 与 Rust 新增 hf response-template parser | 可以减少模型专属 parser，但两个前端的激活、流式工具输出和严格约束边界不同 |
| GLM 工具约束 | 新增非 strict 工具的浅层 structural tag，并对齐 prompt 与 grammar 中的可调用工具集合 | 浅层约束不等于完整 schema；PR 正文对全局严格开关的描述与最终 diff 不一致，按最终代码记录 |
| Parser 迁移 | Step-3.5、MiniMax M3 迁到 streaming parser engine；移除 PyO3 tool-parser bridge | MiniMax 不再依赖 _rust_tool_parser；不能理解为 Rust frontend/parser 整体被移除 |
| 流式与协议 | Responses 标准 reasoning 事件、Harmony 最终 ID 一致性、Anthropic 动态工具块 | 客户端监听旧 reasoning_part 事件需要调整；动态工具块有明确 role 与 reference 限制 |
| grammar / schema | Rust $ref coercion 与 grammar 共用 SchemaRoot；修复深嵌套 schema、spec decode 未约束行 | 防止 schema / decoder / parser 解释不一致；深度上限不是业务对象递归层数的简单映射 |
| 拆分前端 | 修复 render 默认 parser、reasoning 状态透传及 derender prompt 上下文 | render→generate→derender 与耦合 chat endpoint 的一致性更完善 |
| Ascend | 回移 GLM-5.3 always-on reasoning 识别 | 不支持关闭模型思考；目的是避免 reasoning 落入 content |
| Router | 截至抓取时刻无区间已合并 PR 或 main 提交 | 九月 gRPC 主动 tool calling 的限制没有本期代码依据可宣称解除 |

## 3. vLLM：新特性与能力扩展

### 3.1 checkpoint response_template 驱动的 hf parser

| 日期 | 前端/层次 | 能力 | 适用边界 | 来源 |
| --- | --- | --- | --- | --- |
| 10-01 | chat_parsing 基础事件 | 流式 region_open / close 带 delimiter span，开头带 captures；增加 malformed region 事件，错误 region 后还能解析后续内容 | 这是 serving adapter 的基础接口；非流式 parse_response 仍在首个 malformed region 抛出原异常 | [#58603](https://github.com/vllm-project/vllm/pull/58603) |
| 10-01 | Rust hf unified parser | 读取 tokenizer_config.json 的 response_template，用原生 executor 解析 content、reasoning、tools，复用 schema 参数转换 | 显式选择 hf，auto 不自动选；regex 需要可识别有限 literal prefixes，部分 Python regex / template 字段不支持；尚无 template-derived structural grammar | [#59005](https://github.com/vllm-project/vllm/pull/59005) |
| 10-02 | Python hf unified parser | 从运行时 tokenizer 的 response_template 解析工具与 thinking；Chat、Responses、derender 给 parser 传入渲染后的 prompt | 缺失/无效 metadata 会校验失败；不能和其他 reasoning/tool parser 混用，Harmony 使用既有 parser | [#58604](https://github.com/vllm-project/vllm/pull/58604) |

Python 可使用 `--enable-auto-tool-choice --tool-call-parser hf --reasoning-parser hf`；tool 与 reasoning 可分别选用，但不能与其他 parser 混搭。工具 region 关闭且解析成功后整项发出，结尾仍未关闭时尝试解析已有文本，失败则警告并丢弃；unknown tool name 可以被透传。**strict tools、required / named tool choice、parallel_tool_calls=false 暂被拒绝**，因为尚无兼容输出格式的约束 grammar。[PR #58604](https://github.com/vllm-project/vllm/pull/58604)

Rust 显式选择 `--tool-call-parser hf --reasoning-parser hf`。text / reasoning 增量流出；opener 捕获到函数名时可以先发调用开始，参数在 region 关闭后发出。该 PR 没有实现增量参数和 template-derived structural grammar；正文明确 strict / required 在没有 builder 时不获得此格式的约束，**不能套用 Python 的“拒绝”结论**。[PR #59005](https://github.com/vllm-project/vllm/pull/59005)

### 3.2 GLM-4.7 非 strict 工具的浅层 grammar

**10-02：[PR #56403](https://github.com/vllm-project/vllm/pull/56403)** 将 GLM structural-tag builder 纳入本地 registry，按工具 strictness 分别构建：strict 工具保持参数 schema 约束，非 strict 工具采用调用标签、函数名、简单属性 key 和基础 value 类型约束。简单对象允许参数省略、重复和任意顺序；复杂 schema 回退到更宽的原生参数内容。全 strict 请求仍委托 XGrammar builtin。

**以最终 diff 为准的开关行为：** `Glm47MoeModelToolParser.default_tool_strict_level = FUNCTION`，operator 的级别仍为 AUTO 时采用该默认值，因此普通 auto 调用也可获得浅层 tag。operator PARAMETER 可提高所有工具的 schema 约束；`VLLM_ENFORCE_STRICT_TOOL_CALLING=0` **仍禁用 structural tags**。PR 的原始正文描述“在关闭开关时安装 grammar”，已经不能准确概括最后版本；diff 的 `test_glm47_get_structural_tag_disabled_when_flag_off` 明确验证关闭时返回 None。[PR #56403 最终变更](https://github.com/vllm-project/vllm/pull/56403/files)

这延续九月的 function / parameter 区分。浅层 tag 不保证每个必需参数出现，也不等于完整 JSON Schema 校验；“非 strict”不能解释成完全自由输出。后续 #59879 修复了该 builder 与请求工具过滤的组合问题，见第 5 节。

### 3.3 动态工具与延迟加载兼容

| 日期 | 类型 | 变化 | 限制 | 来源 |
| --- | --- | --- | --- | --- |
| 10-02 | Anthropic 兼容扩展/修复 | /v1/messages 接受 system 消息内的 tool_addition / tool_removal，把变更解析进模板看到的工具列表；加入的工具解除 defer_loading，移除的工具隐藏，tool_definition 可声明工具 | 只接受 system role；引用未声明工具返回 400；MCP reference 被拒绝，不表示服务增加 MCP connector | [#57693](https://github.com/vllm-project/vllm/pull/57693) |
| 10-06 | Rust/Python 对齐 | Rust Chat DTO 把工具层和 function 层 defer_loading 传给 HF template，function 层优先，未设置时不输出该字段 | 仅模板可见扩展，不新增服务端 tool search；parser、grammar、choice 校验不因此把 deferred tool 变不可调用 | [#60203](https://github.com/vllm-project/vllm/pull/60203) |

## 4. vLLM：parser / grammar 重构及相关行为变化

| 日期 | 变化 | 用户可见行为或边界 | 来源 |
| --- | --- | --- | --- |
| 10-01 | Step-3.5 / Step-3.7 parser 迁移至 streaming engine，以 Qwen3Parser 小子类处理 XML 工具格式和 thinking | 修复 reasoning 尾空白与当前轮边界判断，参数类型转换随 engine；required / named 原生 XML 路径已由九月 #51810 修复，不能重复归功于本 PR | [#59321](https://github.com/vllm-project/vllm/pull/59321) |
| 10-01 | Rust argument grammar 拆为通用 schema resolution 与模型专属 ArgumentSyntax 渲染 | Kimi K3 参数默认按声明顺序，required 参数必须出现；先解析 $ref 再选择 type 标签；单个含闭合标记的 string enum 分支被丢弃，不再放宽整个 enum | [#59395](https://github.com/vllm-project/vllm/pull/59395) |
| 10-01 | structural-tag snapshot 增加可读树形 outline，与 grammar replay JSON 并存 | test-util 和测试/审阅可读性改进，生产行为不变；不要当成新增解析能力 | [#59393](https://github.com/vllm-project/vllm/pull/59393) |
| 10-02 | 删除只对旧 Transformers 版本有效的兼容代码，包括 HYV4 tool/reasoning 的总为真版本判断 | 前序依赖下限已提高至 5.16.1；本项主要是清理，对支持版本不预期产生行为变化 | [#59762](https://github.com/vllm-project/vllm/pull/59762) |
| 10-05 | batch chat derender 抽取 _parse_and_assemble，共享 kwargs 解析 | 覆盖 reasoning 隐藏、tool ID 与 required/named 空 content 规则，作者明确无行为变化；为 inline derender 后续工作准备 | [#60073](https://github.com/vllm-project/vllm/pull/60073) |
| 10-06 | MiniMax M3 tool parser 从 PyO3 Rust bridge 迁至声明式 parser engine | 正常输出流式/非流式保持一致；残缺或畸形调用保留出错前解析出的参数，而不是旧 bridge 的原文回退/丢掉后续流 | [#59743](https://github.com/vllm-project/vllm/pull/59743) |
| 10-06 | 删除 vllm-tool-parser-py crate、Python RustToolParser adapter、_rust_tool_parser 扩展注册和相应依赖/CI | 最后生产用户 M3 已迁移；保留未来 _rust_* 构建 plumbing。这是 Python bridge 移除，Rust serving/parser 仍存在 | [#59744](https://github.com/vllm-project/vllm/pull/59744) |

**堆叠 PR 核对：** #59744 的 base 是 `bz/minimax-m3-engine`，不是 main。除核对其 merged=true 外，本次还比较 merge commit `b3082d304469437cc38a20efbdb20c3559c8a902...main`：结果 ahead、behind_by=0，说明此 merge commit 已是当前 main 的祖先，移除 bridge 的代码进入了主线；没有仅凭“子分支 PR 已合并”推断主线落地。[PR #59744](https://github.com/vllm-project/vllm/pull/59744)

## 5. vLLM：工具解析与 schema bugfix

| 日期 | 触发问题 | 修复后行为 | 来源 |
| --- | --- | --- | --- |
| 10-01 | Rust Kimi K3 按文本识别 channel marker，reasoning 中普通 BPE 拼出的相同字节误关闭 think | channel/call 边界要求对应专用 token ID 的 attributed input；普通文本 lookalike 保持正文。内部 argument/json 块及 output grammar 仍是文本级，歧义未全部解决 | [#58358](https://github.com/vllm-project/vllm/pull/58358) |
| 10-05 | 非 Harmony prompt 只展示 function tools，但 grammar 取全部 request tools；builtin-only 出现空 tag，namespace function 没有规则 | get_function_tools 成为 prompt、structural tag、旧 required JSON schema 共用来源；namespace 函数按 namespace__name 展平；无可调用函数时 auto 不加 grammar，required/named 返回带 tool_choice 参数的 400。Harmony 显式保留 builtin 能力 | [#59879](https://github.com/vllm-project/vllm/pull/59879) |
| 10-06 | tools parameters 中 $defs=null/list/string 触发 .items 异常，重复定义名称不同 schema 也成为 500 | 用 VLLMValidationError 返回明确 tools 参数 400；正常 definitions 提升路径保持 | [#54850](https://github.com/vllm-project/vllm/pull/54850) |
| 10-06 | Mistral pre-v11 工具 JSON 不是预期 list 时整个请求抛错；string arguments 被重复编码 | 裸 object 当单调用，缺/非 string name 归为空字符串，string arguments 原样保留；无调用的 JSON 回到 content，第二 marker 后内容按 trailing text；流式/非流式共用规则 | [#54844](https://github.com/vllm-project/vllm/pull/54844) |
| 10-07 | Rust grammar 与参数 coercion 分别解释 schema，coercion 忽略 $ref 导致 nested object 成为 JSON string | 共用 SchemaRoot，支持本地 ref、nullable、enum/const、单 schema allOf、类型别名，按实际值递归解析；不可解析/循环/远程引用不网络获取、使用宽松回退。DSML/K3 仍按其 string/type 标签解码 | [#59408](https://github.com/vllm-project/vllm/pull/59408) |

**Kimi K3 的接入要求：** #58358 没有 text-only fallback。没有 token attribution 的直接调用会把 marker 当普通 content；不能只把一个字符串交给 parser 并期待和正式 serving 相同的结果。当前仍有内部参数闭合文本歧义和 text-level grammar 与 token-level parser 不完全一致的已知限制。[PR #58358](https://github.com/vllm-project/vllm/pull/58358)

## 6. vLLM：流式协议与拆分前端 bugfix

### 6.1 Responses 事件与 ID

| 日期 | 问题 → 修复 | 兼容性注意点 | 来源 |
| --- | --- | --- | --- |
| 10-01 | 自定义 response.reasoning_part.added/done 不符合官方 SDK event schema → 使用 response.content_part.added/done，part.type=reasoning_text | 覆盖 Simple/Harmony；是事件名变化，客户端硬编码旧事件必须迁移，PR 的过渡方案只是讨论，没有实现 | [#59652](https://github.com/vllm-project/vllm/pull/59652) |
| 10-04 | Harmony completed 从消息重建时生成新的 item id/call_id → 按每个完成消息的对象身份匹配已流出的记录，复用对应 id/call_id；reasoning 使用 rs_ 前缀 | 内容和 incomplete 状态仍由 rebuild 得出；从未流出的 zero-delta 项可新建 ID。消息级匹配避免 builtin python 在重建类型变化时偷取下一 reasoning ID | [#59859](https://github.com/vllm-project/vllm/pull/59859) |

九月 [#59307](https://github.com/vllm-project/vllm/pull/59307) 处理非 Harmony 的流式最终响应；本期 #59859 补的是 Harmony 分支，不能把两者混为同一修复。

### 6.2 render → generate → derender 一致性

| 日期 | 问题 → 修复 | 来源 |
| --- | --- | --- |
| 10-02 | SentencePiece derender 没有 prompt decode context，首 token 的前导空格丢失，batch 中截断 UTF-8 会补乱码 → 用 prompt tail 初始化 decode window；可从请求 prompt_token_ids 或 generate response 取值，缺失时保留旧行为并警告；统一增量 decode 和特殊 token 选项 | [#59046](https://github.com/vllm-project/vllm/pull/59046) |
| 10-05 | scale-out render→generate 丢掉 reasoning_ended 和 reasoning_parser_kwargs，结构化约束启动时机与 chat 不同 → 渲染结果和 generate 透传两字段，include_reasoning 与 template kwargs 保持 engine/frontend 一致 | [#60062](https://github.com/vllm-project/vllm/pull/60062) |
| 10-06 | GPU-less render launcher 使用原始空 reasoning CLI flag，忽略模型默认 parser，gpt-oss derender 返回原始 Harmony markup → 共用 structured_outputs config 构建，render/derender 使用解析后的配置值，明确 flag 优先 | [#54835](https://github.com/vllm-project/vllm/pull/54835) |

#54835 正文保留了“等待 parser-aware streaming derender”的早期上下文；九月 [#50550](https://github.com/vllm-project/vllm/pull/50550) 已合并。因此不能把正文的历史 501 描述当成本期独立新限制；这里仅记录默认 parser 配置修复，不额外宣称每种模型流式组合已验证支持。

### 6.3 相邻响应序列化修复

**10-07：[PR #59072](https://github.com/vllm-project/vllm/pull/59072)** `/cohere/v2/chat` 非流式请求 `logprobs=true` 时此前已计算但未返回 logprobs；现在携带 token IDs，并映射为 Cohere LogprobItem。流式保持原状，需要额外处理 parser 缓冲或转为 tool call 的 token。这属于相邻前端输出修复，不是新增工具执行能力。

## 7. vLLM：结构化输出的共享正确性与稳定性修复

| 日期 | 问题 | 修复与边界 | 来源 |
| --- | --- | --- | --- |
| 10-01 | spec decode 的 draft 未到 scheduler 时，padding 后的 grammar rows 是 full mask，worker 可接受根本未被约束的 draft | 每请求记录实际被约束的 leading draft 数，超出位置作无效 draft 后走目标分布重采样，阻止采样进入无约束行；handoff 身份丢失的根因是另一个改动，不把本项说成全链路修完 | [#54442](https://github.com/vllm-project/vllm/pull/54442) |
| 10-06 | 深嵌套 JSON schema / structural tag 递归抛 500，或同步转换阻塞 API event loop 与 health | 增加最多 64 层 JSON object/array 容器嵌套检查，转换前返回 400；限制计算容器层数，工具 wrapper 也影响实际校验形状，不等价于“最多 64 层业务属性” | [#60036](https://github.com/vllm-project/vllm/pull/60036) |

上述共享 grammar 路径也可能影响 schema-constrained tool calling，所以列为主题相关修复。#60036 作者记录深 structural tag 的旧版 health 阻塞约 33.8s、新版及时 400；这只是其 A/B 请求与硬件环境的测量，不是本次复现实验。[PR #60036](https://github.com/vllm-project/vllm/pull/60036)

## 8. vLLM Ascend：GLM-5.3 reasoning 回移

**10-05，合并至 main：[PR #17876](https://github.com/vllm-project/vllm-ascend/pull/17876)** 回移上游 [vLLM #56994](https://github.com/vllm-project/vllm/pull/56994) 的模板检测与 constructor 修正。GLM-5.3-Flash 总会 reasoning，客户端仍传 `thinking=false` / `enable_thinking=false` 时，旧 parser 关闭 extraction，scratchpad 落入 content；patch 匹配 always-on 模板签名后忽略这些不支持的关闭参数，让 reasoning 与答案分离。

patch 在 kwargs 副本上归一化，保留其余配置与 constructor 调用约定；检测到上游已有 helper 就跳过，也避免重复包装。旧模板和本身支持开关的模板保留行为。作者报告目标容器实际 tokenizer 的 12/12 离线 parser A/B 从泄漏变为不泄漏，另有 2/2 正常请求控制通过；这不是本次运行 NPU 模型的结果。[PR #17876](https://github.com/vllm-project/vllm-ascend/pull/17876)

**本项不提供关闭模型思考的能力。** 来源 vLLM #56994 的修复不按本期新增上游 PR 重复计数；本期落地事件是 Ascend patch 的合并。当前支持版本何时可以移除 patch，需要逐一确认上游 pin。

本期 Ascend 其他 PR 主要涉及 KV cache、PCP/DCP、算子、MoE、测试与部署文档。模型内部路由及 KV 传输优化没有被写成 vllm-project/router 的新特性；本表也不声称 Ascend 各版本同步获得本期全部 vLLM main parser 变化。

## 9. vLLM Router：本区间无已落地主线变更

| 核对项 | 截至抓取时刻的结果 | 来源 |
| --- | --- | --- |
| 10.01–10.07 的已合并 PR | 0，REST search incomplete_results=false | [PR 检索](https://github.com/vllm-project/router/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-10-01..2026-10-07) |
| main 的区间 commits | commits API 返回空数组 []，无直接提交可补充 | [区间 commits API](https://api.github.com/repos/vllm-project/router/commits?since=2026-10-01T00%3A00%3A00Z&until=2026-10-07T23%3A59%3A59Z&per_page=100) |
| main 最新提交 | 10340964，UTC 2026-09-30 14:33:18，更新社区二维码，属于上期 | [提交](https://github.com/vllm-project/router/commit/10340964bd8c64a25fcd36458c43edf924dfe2e0) |

因此本期没有可归入新特性、bugfix、性能、重构或 CI/文档维护的 Router 主线落地项。开放 PR、其他分支 push、repository updated_at 不等价于 main 代码变更。

九月 [Router #283](https://github.com/vllm-project/router/pull/283) 的 gRPC adapter 主动 tool calling 限制，本期没有代码变更证据表明解除；vLLM 的 hf parser 或 PyO3 bridge 移除也不自动改变这个 adapter。九月 WASM、Program Progress-TTL、NIXL push、L0 cache 不在本期重复计数。

## 10. 建议的代码阅读顺序

1. **模板驱动解析**：#58603 的 region events → #58604 的 Python serving adapter → #59005 的 Rust unified parser，比较 prompt anchor、流式工具发出时机与请求约束。
2. **约束与 prompt 对齐**：先读 #56403 最终 builder、default_tool_strict_level 和 flag-off 测试，再读 #59879 的 get_function_tools。对照九月 strict-level 记录理解浅层与参数约束的差异。
3. **schema 类型一致性**：#59395 的 ArgumentSyntax 与 schema options → #59408 的 SchemaRoot/coercion，追踪 Pydantic-style $ref、enum 和 nullable 的处理。
4. **流式协议**：#59652 的事件名 → #59859 的 per-message streamed ID 记录，检查 reasoning、function_call、built-in tool、incomplete 与 zero-delta 分支。
5. **parser 迁移与桥接**：#59321 / #59743 的 engine adapters → #59744 的 bridge 删除。不要把 Python extension 和独立 Rust frontend 混同。
6. **token-in/token-out 一致性**：#54835 的默认 config → #60062 的生成状态透传 → #59046 的 decode context → #60073 的 parse/assemble 抽取。
7. **共享 grammar 正确性**：#54442 的 constrained draft 数与 rejection sampler，再看 #60036 的 JSON 容器深度校验。

以上是本次阅读建议。具体源码从各 PR 的 Files changed / merge commit 查看，避免用不断变化的 main 页面替代当时 diff。

## 11. 定时执行与后续补齐

- 每周四 **北京时间 20:00（Asia/Shanghai）** 在本聊天执行，统计前一天结束的七个日期，继续采用 UTC 合并日期口径；例如 10 月 8 日正式覆盖 10.01–10.07，10 月 15 日覆盖 10.08–10.14。
- 按 `toolcall和router变更-MMDD-MMDD.md` 保存；标题保留年份，跨年/同名年份冲突时文件名也增加年份。
- 文档校验后，用 `Liccol <740821011@qq.com>` 提交本次记录并推送 `origin/main`。不把凭证写入文件，不对远程冲突使用 force push。
- 已完整成功统计且已推送、来源无实质变化时不重复通知；暂定报告必须在区间结束后复查。失败时保留本地成果并报告具体阶段。
- **本次区间状态：暂定，等待 2026-10-08 定时复查补齐。** 这里的状态描述统计完整性，不代替 Git 推送结果；推送成功与否由执行结果另行报告。
