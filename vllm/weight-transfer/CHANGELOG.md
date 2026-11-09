# Packed tensor 文档版本记录

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
