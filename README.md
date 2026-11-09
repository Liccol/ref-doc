# ref-doc

- [Tool Calling 变更记录](vllm/toolcall/)：vLLM 与 vLLM Ascend 中与工具调用相关的新特性、变更和问题修复。
- [Router 变更记录](vllm/router/)：vLLM Router 与 SGLang Model Gateway 服务路由器的新特性、变更和问题修复。具体范围见 [Router 调研说明](vllm/router/README.md)。
- [MiniMax Agent 调研](minimax/agent/)：五个相关仓库的内容总览与变更记录，重点关注 minimax-code。

各主题记录按时间区间分别存放、独立维护。每周四北京时间 20:00 生成过去一周的本地周报，人工审核后再提交。

- [MiniMax-M3.1-Flash 模型结构与 A5 开箱验证](minimax/minimax-m31-flash/)：全层精度、RL 测试命令与结果、减层待补验范围。

- [Weight transfer 设计文档](vllm/weight-transfer/)：Packed tensor 方法流程、Ascend 适配与复用方案，附文档版本记录。

## 环境信息维护约定

文档、历史 Markdown、命令样例、结果 JSON 和压缩附件均不得包含实际验收地址、账号口令、容器名或内部路径。公开软件版本、通用硬件类型和测试结论可以保留。网络示例使用文档专用地址，路径和容器使用占位示例，执行前须自行配置。

当前文件及可达 Git 历史均做环境脱敏，保留逐次提交记录和顺序；脱敏导致提交 SHA 变化。独立历史 Markdown 同样保留脱敏后的版本。
