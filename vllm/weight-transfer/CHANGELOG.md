# Packed tensor 文档版本记录

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
