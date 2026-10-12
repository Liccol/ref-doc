# MiniMax Agent 调研

本目录保存五个 MiniMax 相关仓库的内容总览与后续变更记录，重点关注 minimax-code。

| 仓库 | 职责 | 总览 |
| --- | --- | --- |
| minimax-code | 终端编码 Agent、TUI、exec、ACP 和本地运行时；重点跟踪 | [仓库总览](minimax-code/仓库总览.md)、[项目架构](minimax-code/项目架构.md) |
| Mini-Agent | Python 单 Agent 示例、工具循环、上下文与 MCP/Skills | [仓库总览](Mini-Agent/仓库总览.md) |
| cli | mmx 多模态平台命令、SDK 和 Agent 配置辅助 | [仓库总览](cli/仓库总览.md) |
| MiniMax-MCP | 向 MCP 客户端暴露音频、图像和视频工具 | [仓库总览](MiniMax-MCP/仓库总览.md) |
| OpenRoom | 浏览器桌面、应用 Action 与 Vibe 开发流程 | [仓库总览](OpenRoom/仓库总览.md) |

总览是 2026 年 10 月 12 日核查的源码快照，标注提交与来源。它们不是过去一周的变更周报，也不代表已验证线上全部能力。

后续变更分别保存在对应仓库目录，命名为 `变更-MMDD-MMDD.md`，标题包含年份，跨年或重名时文件名增加年份。跟踪 main 已落地的新特性、行为变化、兼容性及问题修复；minimax-code 重点展开执行链路、工具、会话、模型接入和权限变化，其余仓库保持紧凑。

沿用每周四北京时间 20:00、截至周三的七个完整 UTC 日期及 PR 合并时间口径。无 PR 的直接提交单独说明时间与提交来源。与 vLLM Tool Calling / Router 分开保存，人工审核后再提交推送。
