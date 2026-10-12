# Router 调研范围

本目录跟踪两套服务路由器的 main 分支：

- [vLLM Router](https://github.com/vllm-project/router)：独立仓库。
- [SGLang Model Gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway)：SGLang 仓库中的 `sgl-model-gateway/` 目录。

每期周报按 `vLLM Router` 和 `SGLang Model Gateway` 分节，分别记录新特性、行为变更和问题修复；某一项目没有相关变更时单独说明。重点变更写明用途、启用方式及适用边界，普通变更使用日期、变更说明和 PR 链接组成的紧凑表格。

SGLang 按 `sgl-model-gateway/` 的实际文件变更筛选 PR；目录外的接口、配置或依赖改动，只有明确影响该 Gateway 时才纳入。不将 SGLang 全仓变更、模型推理内核或 MoE 专家路由混入服务路由器记录。

每周四北京时间 20:00，总结截至周三的七个完整 UTC 日期，以 PR 合并时间统计。周报仍保存为 `router变更-MMDD-MMDD.md`，标题包含年份，跨年或同名冲突时文件名增加年份。生成后人工审核，再按明确指令提交推送。

SGLang 调研范围自 2026 年 10 月 12 日起加入；已有九月及 10 月 1 日至 7 日记录仍只覆盖 vLLM Router，尚未补查同期 SGLang 变更。Tool Calling 的调研范围和内容保持不变。
