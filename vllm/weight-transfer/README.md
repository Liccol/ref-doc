# Weight transfer 设计文档

- [Packed tensor 上游复用与 NPU 适配设计](packed-tensor-design.md)：当前版本 **v1.0.24**，逐一对照 HCCL/IPC producer/consumer 重构前时序，并在图后说明差异适配和复用结论。
- [双配套验证说明](packed-tensor-validation.md)：记录当前 PR 对 vLLM v0.33.0 和固定 main 的 UT/E2E 结果、配置及限制，不含部署定位信息。
- [文档版本记录](CHANGELOG.md)：记录每次修改的版本及内容。
- [历史快照](versions/README.md)：保留各阶段原有设计结论，不作为当前实现或测试状态。

## 阅读顺序与目录结构

先阅读设计文档第 1 章了解目标；重点评审第 3、4 章的四组方法时序对照及图后适配说明，再结合第 2 章整体结构理解当前实现；固定源码基准和测试证据见验证说明。历史方案只在追溯时查阅，避免与当前实现混用。

拟议的上游修改见设计文档第 7 章，与当前已实现行为及已通过测试的范围分开记录。

```text
weight-transfer/
├── README.md                      # 阅读入口与维护约定
├── packed-tensor-design.md         # 当前设计与适配边界
├── packed-tensor-validation.md     # 当前配套验证证据
├── CHANGELOG.md                    # 文档版本记录
└── versions/                      # 历史设计快照
```

## 版本维护约定

1. 首次入库版本为 `v1.0.0`；后续每次修改至少递增修订号，例如 `v1.0.1`、`v1.0.2`。
2. 每次修改同步更新受影响文档顶部版本号、本页当前版本和 CHANGELOG，不能覆盖历史版本记录。设计和验证说明使用同一文档修订号。
3. 每次发布保留独立 Git commit，提交信息包含文档版本号；不 squash、rebase 或 force push 已发布的文档历史。
4. 版本表示文档修订，不表示 vLLM 或 vLLM Ascend 软件版本。源码配套版本在文档中另行记录。
5. `packed-tensor-design.md` 保持为当前版本；历史 Markdown 保存在 [versions](versions/README.md)，每次归档先移除验收环境定位信息。历史快照保留原文档版本号，并标明脱敏处理。
