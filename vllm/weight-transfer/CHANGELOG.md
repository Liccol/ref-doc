# Packed tensor 文档版本记录

## v1.0.15

- 新增 4.7 节“CUDA 操作穷举与最终继承边界”：对上游 packed_tensor.py 全部 CUDA 相关操作逐行穷举，分三类——A 类已解决（wrapper/等价 API/真机验证）、B 类 ascend 侧可解决（torch_npu 自动 patch reduce_tensor；**修正**：rebuild_cuda_tensor 经 importer 类继承覆写可解，PackedIpcChunk 非 CUDA 障碍）、C 类无法解决仅 2 处（producer L339 device="cuda" 字面量、consumer L468 get_device_properties().uuid）。
- 同步修正 4.3/4.4/4.5：producer 硬差异 3→1（句柄导出改判适配、协议改判选型）、consumer 硬差异 3→1（重建改判可注入）、总览图与总结论同步更新。
- 结论：两个残留合计约 3 行上游参数化改动（device 参数化 + uuid 注入），合入后 IPC 两方法可改为直调继承（consumer 需 NPUBufferImporter 覆写 rebuild）；合入前维持自写不变。
- 保留 v1.0.14 快照。

## v1.0.14

- 4.6 节补充 `torch.cuda.get_device_properties` 的仓内替换先例核查：全仓 git 历史（`-S`）从未有人写过或重绑过该调用；替代先例走 vLLM Platform 扩展点（`NPUPlatform.get_device_uuid` 调 `torch.npu.get_device_properties().uuid`、`num_compute_units` 映射 cube_core_num）。
- 新增"注入时的选型注意"：engine 自建 `npu_generate_uuid` 而非用 platform `get_device_uuid`，因 props.uuid 在部分 torch_npu/CANN 版本不可靠（platform 有 RuntimeError 防御）；将来上游注入 uuid 查找时应选 `npu_generate_uuid`。
- 保留 v1.0.13 快照。

## v1.0.13

- 新增 4.6 节“IPC cuda 绑定的可解性与继承路径”：核对仓内 `npu_ipc_engine.py` 已有的同类解法（`npu_generate_uuid` 替代 CUDA UUID、`rebuild_npu_tensor` + device_index 覆写替代 `rebuild_cuda_tensor`、torch_npu 自动 patch `reduce_tensor`、`torch_cuda_wrapper` 重绑 `torch.cuda.*`）。
- 说明函数级直调上游 IPC 仍差的三个口子：producer `device="cuda"` 字面量、consumer 按名导入的 `rebuild_cuda_tensor`（monkey patch 不可回头）、wrapper 未重绑 `get_device_properties`。
- 继承路径建议：本 PR 维持自写 IPC（解法已同构落在 engine 层）；中期向上游提 2 个极小 PR（device 参数化、rebuild/UUID 可注入），合入后与 HCCL 路径同样改为委托继承。
- 修正 4.4 节 UUID 行表述：`torch.npu.get_device_properties` 存在（platform.py 在用），选型 `npu_generate_uuid` 是设计决策而非能力缺失。
- 保留 v1.0.12 快照。

## v1.0.12

- 新增第 5 章“packed 相关未合入 PR 的影响分析”：检索 vllm-ascend 全部 7 个 packed weight transfer 未合入 PR（#20740/#18078/#16999/#19127/#12713/#18763/#20614），逐个判定继承重构完成后是否还需要修改。
- 结论：#16999 与 #19127 被继承取代（可关闭或改写为可选加固）；#18078 与本 PR 直接冲突需重大返工重评；#20740 不受影响仅需 rebase；#12713 仍需要但补丁位置要改到新入口；两个 v0.30.0 release 分支 PR 不受影响。附合入顺序建议。
- 保留 v1.0.11 快照。

## v1.0.11

- 新增第 4 章“四个方法逐环节对照”：HCCL producer/consumer 与 IPC producer/consumer 按“环节 | vLLM | vllm-ascend | 差异性质（一致/适配/硬差异）”逐环节对比，每节给出继承结论，4.5 节以 mermaid 总览图汇总硬差异数量。
- 结论：HCCL 两方法硬差异为 0 → 委托继承；IPC 两方法各 3 处 CUDA 专属硬差异（device 字符串、rebuild 函数、UUID 解析/协议类型）→ 本地自写、可共享环节复用上游。
- 源码依据：vLLM main 24420b7a 源码逐行核对（producer L146-193、consumer L196-296、IPC L313-491）；保留 v1.0.10 快照。

## v1.0.10

- 新增 3.1 继承关系图（mermaid）：以函数级委托/复用箭头呈现两侧方法关系，明确标注 4 条继承路径（HCCL producer/consumer 委托、IPC consumer 复用解包、_hccl_group 复用 stream 缓存）、上游 IPC 未继承区与 torch_cuda_wrapper 适配关系；明确本文档不涉及类继承。
- 保留 v1.0.9 独立 Markdown 快照。

## v1.0.9

- 结构重写：按“背景与继承原则 → 方法清单对照 → 逐方法继承判定”三节组织，取代原五章节流程详解；主文档从 829 行精简至约 110 行。
- 新增上游 11 个符号与 Ascend 侧符号的完整清单表，以及逐方法继承判定表：6 个完全/间接继承、2 个委托继承（含 NPU 适配包装）、3 个 IPC 相关符号因 CUDA 专属调用（`device="cuda"` 硬编码、`rebuild_cuda_tensor`、`torch.cuda.get_device_properties`）或跨进程协议差异无法直接继承，改为 2 个自写函数 + dict 协议替代。
- 源码基准更新为 260e4c3c3；真机验证结论、跨 chunk 机制分析、全部流程图保留在 v1.0.8 快照中，主文档以链接引用。
- 保留 v1.0.8 独立 Markdown 快照（流程详解终版）。

## v1.0.8

- wrapper 语义文档化：`torch_cuda_wrapper` 补充 docstring，注明进程级重绑定、退出不恢复，避免被当作通用可逆 context manager 复用（检视 P1-4 收尾；代码 260e4c3c3，无行为变化）。图 2A 节点、2.4 作用域说明、4.1 文件边界表同步更新。
- 5.4 改为两阶段验证结构：第一阶段构造性时序探针（同 stream copy 模拟 HCCL）的 10/10 错误结论明确限定为“模拟环境、人为延迟时序”，不适用于真实 HCCL。
- 5.4 补记第二阶段真机最小用例（773c8947e，2×Ascend910C A3、CANN 9.2.0、torch/torch_npu 2.10.0，2 rank 真实 HCCL）：accelerator C 级 API / Event / record_stream / 跨 stream copy 真机 gate 通过；端到端数据校验通过；跨 chunk 读序缺口在真实 HCCL 时序下未复现（2MB 与 16MB chunk 各 3/3 正确），此前的“读序无保证”结论修正为“真实时序未复现、机制上仍无显式事件保证”。
- wait_event 防御对照通过（无死锁无回归），留作可选加固并注明 generic `torch.Stream` 与 `npu.Stream` 混用会静默失效；未写入生产代码。
- 5.5 表格同步：stream 缓存键的真机前提已验证；新增“wrapper 语义文档化”条目；clone 读序修复的后续讨论边界更新。保留 v1.0.7 独立 Markdown 快照。

## v1.0.7

- 设计变更：删除 owning-unpack hook（进程级模块属性重绑定），独立存储改由 HCCL 侧 `_owning_post_unpack` 回调包装与 IPC 侧本地 clone 提供；先 unpack 得到 views、再逐项 clone，未对齐 offset 由原 unpack 的 contiguous 拷贝处理。
- drain 委托：本地 finally 仅在异常路径执行，正常路径交给上游原 producer/consumer 的结尾 drain，消除重复同步。
- stream 缓存键与上游统一为 `torch.accelerator.current_device_index`，消除与原循环键分叉导致第二套缓存 / `streams.index` 失败的隐患。
- negative-shape 校验抽为 `_reject_negative_shape`，HCCL 与 IPC 共用。
- 图 2A/2B、3A、3B、4、5、6、附图 A 按新实现重绘或修订；图 6A/6B 合并为单张图 6，删除随 hook 一并失效的符号重绑定状态图。
- 5.4 机制表述修正：raw buffer 的 record_stream（含 caller stream）与 receive-ready 链在两套配套均已存在，保留的验证边界是 clone 的读序，而非“完全无保护”；探针基于 v1.0.6 hook 版本，合并前需按 773c8947e 重做真机最小用例。
- 源码基准更新为 773c8947e；新增上游符号签名/正常 drain 护栏 UT、缓存键用例与 `_HCCLPackedGroup` 直接用例。保留 v1.0.6 独立 Markdown 快照。

## v1.0.6

- 依据 e5b128e35 代码核对修正图 3A：finally drain 的发起者改为 `_hccl_group`，补充 group 包装登记每 slot 最后通信 buffer 的步骤，标注 EOF 正常 drain 与 finally drain 在正常路径重复同步。
- 图 5 重画：HCCL 与 IPC 的 clone 节点拆分为两条链各自持有，删除与文字矛盾的合并节点；未修复的"回收复用与异步读"冲突改为双入边冲突节点表达。
- 图 6 拆为 6A 静态结构与 6B 符号重绑定状态图：模块属性 `unpack_tensor` 标注为可重绑定，安装前/安装后分离，消除双向依赖环引发的"递归"误读。
- 图 2B 与图 2A 节点命名对齐，标注先幂等安装、后调用的顺序；附图 A 补充 logger.error 记录步骤。
- 仅修正图表与对应文字，章节结论、源码基准和验证记录不变；保留 v1.0.5 独立 Markdown 快照。

## v1.0.5

- 增加 UML 静态结构图，列出两侧模块函数、PackedChunk、HCCL 通信包装类和 owning-unpack 闭包。
- 标明直接委托、运行时接口依赖、模块 hook 与原函数调用，说明没有新增 packed 类继承。
- 明确未复用的 vLLM IPC 方法和 importer；保留 v1.0.4 独立脱敏 Markdown。
- 仅修改文档，源码与测试结论不变。

## v1.0.4

- 增加重构前后结构图、函数复用关系图、HCCL 发送与接收时序图、IPC 时序图、存储生命周期图及异常清理附图。
- 明确函数委托与类继承、发送与接收 stream、Python 返回与设备完成的区别。
- 标注 wrapper / owning hook 的实际作用域，以及尚未修复的 clone 跨 stream 就绪和生命周期边界。
- 说明 IPC 共享 buffer 的复用前提与既有 trainer 同步范围，不把 RPC 返回当成 worker 设备完成证明。
- 保留 v1.0.3 独立脱敏 Markdown 快照；源码和验证结论不变，未新增硬件测试。

## v1.0.3

- 移除验收机器编号、物理设备编号和内部日志路径，保留测试结论与验证范围。
- 增加 v1.0.0、v1.0.1（阶段草稿）、v1.0.2 的独立脱敏 Markdown 快照及目录。
- 历史快照与当前文档均不包含实际部署地址、容器名、账号或内部路径；按授权同步脱敏 Git 历史，保留逐次提交。

## v1.0.2

基于代码提交 `e5b128e355ef8b88fad4031bdcab28beb1035157` 更新。

- 配套更新为 vLLM main 24420b7a 与 v0.33.0 4ba787ec，更新 main 合并基准及固定源码链接。
- 将 IPC owning-unpack 与 producer callback 校验共享从“建议”改为已落地，明确 NPU 分包、重建和同步仍保留本地。
- 更新 v0.33.0 与 main 接收循环差异，保留 V2 wrapper 的 NPU-only、进程级且不恢复行为说明。
- 记录两套 CPU 回归各 179 passed、独立 IPC 用例 1 passed，以及 E2E 仅完成 collection 的范围。
- 增加真实 NPU 解包独立性与跨 chunk 机制探针对照，明确当前仍未修复跨流就绪和 clone 生命周期问题。
- 保留 v1.0.1 阶段草稿独立提交，不覆盖历史记录。

## v1.0.1（阶段草稿归档）

- 按 NPU-only 约束，改为复用 V2 的 torch_cuda_wrapper，删除 packed 模块的自定义 API 代理。
- 明确现有 wrapper 修改进程级 torch.cuda，且退出后不恢复。
- 更新两套配套的测试范围和日志；IPC 保持 main 原样，不新增跨 chunk event。
- 代码与文档均不提交、不推送，等待本地评审。

## v1.0.0

首次入库，建立文档版本记录。

- 列出 vLLM packed 模块的 7 个函数、importer 的 3 个方法及两个 chunk 数据结构。
- 补充方法流程概览、广播和 IPC 调用关系图。
- 按方法对照 vLLM 流程、当前 Ascend 流程、差异、复用方案和保留边界。
- 明确 HCCL 循环复用、NPU IPC 保持 main 原样，以及公共 owning-unpack helper 尚未实施。
- 记录 main/v0.32.0 固定配套、已有单测结果与硬件验证限制。

源码状态：代码分支 `packed-vllm-loop-reuse` 的 `c2d66e163` 加文档所述未提交工作区修改。文档发布不代表代码已提交或推送。
