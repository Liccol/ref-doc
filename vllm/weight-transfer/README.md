# Weight transfer 设计文档

- [Packed tensor 方法、流程与复用设计](packed-tensor-design.md)：当前版本 **v1.0.4**。
- [文档版本记录](CHANGELOG.md)：记录每次修改的版本及内容。

## 版本维护约定

1. 首次入库版本为 `v1.0.0`；后续每次修改至少递增修订号，例如 `v1.0.1`、`v1.0.2`。
2. 每次修改同步更新文档顶部版本号、本页当前版本和 CHANGELOG，不能覆盖历史版本记录。
3. 每次发布保留独立 Git commit，提交信息包含文档版本号；不 squash、rebase 或 force push 已发布的文档历史。
4. 版本表示文档修订，不表示 vLLM 或 vLLM Ascend 软件版本。源码配套版本在文档中另行记录。
5. `packed-tensor-design.md` 保持为当前版本；历史 Markdown 保存在 [versions](versions/README.md)，每次归档先移除验收环境定位信息。历史快照保留原文档版本号，并标明脱敏处理。
