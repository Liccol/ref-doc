# 结构详解的官方来源快照

获取日期：2026-11-12。模型：[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/ce7f4df87d2e05262f5775f373dac4772f134041)。所有文件固定 revision `ce7f4df87d2e05262f5775f373dac4772f134041`。

| 文件 | 内容 |
| --- | --- |
| [config.json](config.json) | 官方主模型原始配置 |
| [preprocessor_config.json](preprocessor_config.json) | 官方图像/视频处理器配置 |
| [dspark-config.json](dspark-config.json) | 官方 `dspark/config.json` 原始内容 |
| [weight-header-samples.json](weight-header-samples.json) | 官方主模型各选定 safetensors shard 的元数据抽样，共 73 项，保留 dtype、shape 和所属 shard |

Header 抽样通过官方 `model.safetensors.index.json` 定位文件；对选定 shard 发起 HTTP Range 请求，读取开头 8-byte little-endian header 长度及后续 JSON，再关闭响应。未加载 tensor 数据区，未执行远程 Python 模型代码。抽样覆盖 Embedding、LM Head、层 0 attention/Dense、层 3 router/首个路由专家/共享专家、视觉第 0 层和融合 MLP；不是全量权重校验。

本文档分析和快照不更新历史 NPU 验收结论。公式与各 shape 的解释见[前向详解](../../model_analyze.md)。
