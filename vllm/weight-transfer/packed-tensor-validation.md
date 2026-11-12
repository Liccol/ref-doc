# Packed tensor 双配套验证说明

文档版本：**v1.0.19** · 测试日期：2026-11-11 · 更新日期：2026-11-12

本文记录 [PR #20676](https://github.com/vllm-project/vllm-ascend/pull/20676) 的已完成验证，配合 [设计文档](packed-tensor-design.md) 阅读。不包含独立问题修复分支的代码或测试结果。

## 1. 源码与运行前核对

Ascend 固定为 `packed-vllm-loop-reuse` 分支、提交 `870b61a90f299d10728b94080089eb5ab5095c4b`。

| 配套 | vLLM 提交 | 实际导入版本 |
| --- | --- | --- |
| tag v0.33.0 | `4ba787ec9959e566d3b587a796dff07673324da5` | `0.33.0` |
| main | `24420b7a341af37a9bc3186e6fc6062512e0aa58` | `0.1.dev1+g24420b7a3` |

切换 vLLM 后重新执行 editable 安装。每次选定的 pytest 调用前，检查两侧 HEAD、Ascend 分支、受版本控制源码的工作区状态、实际导入模块和版本，以及修改文件 SHA256；只允许生成的未跟踪运行产物。

| 文件 | SHA256 |
| --- | --- |
| `vllm_ascend/distributed/weight_transfer/packed_tensor.py` | `0610f16091cc6fe891b5ce468de2faa8516e2cd27f843cf6629d66b9c8c02c19` |
| `tests/ut/distributed/weight_transfer/test_packed_tensor.py` | `0b96637dc11776535bce2d44c55c4def38a82c28bbd278f15658ce7f82ff70d1` |

## 2. UT：每套 186 项通过

```bash
python -m pytest -q \
  tests/ut/distributed/weight_transfer \
  tests/ut/worker/v2/test_v2_utils.py
```

测试采用 CPU tensor 和 mocked NPU API。验证时设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，并通过临时 `npu-smi` shim 选择仓库的 CPU mock 初始化路径，未改仓内测试源码。

覆盖内容包括：

- scalar、空 tensor、混合 dtype、未对齐 offset、独立存储与回调契约。
- IPC 元数据长度、dtype、负 shape、字节数与容量校验。
- native HCCL stream 转发、异常 drain、失败 storage 保留及异常优先级。
- 正常路径 producer/consumer 分别使用 1 和 3 个 slot，断言最后一次 stream 上下文退出后 drain 每个 slot，包含未使用 slot。

UT 中的 stream mock 不构成真实设备时序证明。

## 3. E2E：每套五项通过

用例目录为 `tests/e2e/pull_request/two_card/rlhf/weight_transfer`。每个用例单独运行；packed 用例默认存在 skip，验证通过临时外部 `force_packed` pytest 插件仅移除该 skip，让用例实际执行。没有修改生产代码或仓内测试文件。

| 用例 | tag v0.33.0 | main |
| --- | --- | --- |
| Qwen HCCL unpacked | 通过 | 通过 |
| Qwen HCCL packed | 通过 | 通过 |
| GLM IPC unpacked | 通过 | 通过 |
| Qwen IPC unpacked | 通过 | 通过 |
| Qwen IPC packed | 通过 | 通过 |
| 合计 | 5/5 | 5/5 |

每个选定的最终运行均有退出码 0 和完成的 JUnit 报告；未完成或中断的尝试不计通过。目录中的五个用例全部执行，并非只做 collection。

### 3.1 模型与执行配置

| 项目 | 配置 |
| --- | --- |
| Qwen | Qwen3.5-35B-A3B，裁为 4 层 |
| GLM | GLM-5.1，裁为 4 层、8 个 routed experts |
| 通用配置 | BF16、`weight_nz_mode=0`、`FULL_DECODE_ONLY`、capture size `[1]` |
| HCCL | TP1 推理服务与独立 trainer，真实多进程 HCCL 通信 |
| IPC | TP2、trainer 与 worker 同卡部署，真实多进程 NPU IPC |
| 断言 | 启动加载的参考结果与两轮在线权重更新比较，检查生成文本和 logprobs |
| 随机权重 | 仓库 fixture 生成，固定 seed `20260915` |

使用离线 tokenizer/config 资源及 dummy 权重。GLM 使用官方未量化 config，资源版本为 `26e1bd6e011feb778d25ae34b09b07074139d92d`；测试隔离缓存不带本地量化描述或量化权重索引。

### 3.2 公开软硬件信息

| 组件 | 版本或类型 |
| --- | --- |
| 设备 | 两张 Ascend 910 NPU |
| 驱动 | `26.0.rc1` |
| CANN | `9.2.1` |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post7` |

初次 HCCL packed 初始化遇到 NPU socket 端口占用，测试环境设置 `HCCL_NPU_SOCKET_PORT_RANGE=auto` 后重跑通过；该设置未加入生产代码。机器中断后也重新核对源码并补跑，最终 main packed IPC 为 `1 passed, 1 deselected`、退出码 0。这里的 deselected 是同文件的 unpacked 用例，已在另一次选定运行中通过。

## 4. 结论与限制

本次证明纯继承 PR 在所列两套配套、裁层模型和 graph 配置下通过受影响 UT 及五项真实 NPU E2E。

仍未覆盖或不能据此推断的范围：

- 全深度模型、吞吐/延迟、显存长期稳定性和压力测试。
- 任意 loader 自有 stream 或构造性延迟下的 clone 读序与 allocator 生命周期安全性。
- 多 chunk IPC 长期引用计数配对与无泄漏。成功运行仍有进程退出时共享 tensor 释放警告，原始日志保留了该信息。
- 完整硬件 CI 或全部 pre-commit 通过。此前 Ruff、语法、diff 和文档 Markdown 检查通过，但 `bash format.sh ci` 的部分 hook 受本地 shell/path 或缺失 shellcheck 阻塞。

独立的跨 stream 问题修复后续在单独分支处理。本次更新只整理已有测试证据，没有重新执行硬件测试。原始 preflight、命令、日志、退出码和 JUnit 单独保留；本目录不收录包含部署定位信息的原始附件。
