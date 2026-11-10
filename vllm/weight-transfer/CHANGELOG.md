# Packed tensor 文档版本记录

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
