# 关闭 prefix cache 的逐 DP 补测

双机 TP4 DP4 EP，BF16，FULL_DECODE_ONLY，上下文 131072。每组串行请求固定 DP0–3，每个 DP 覆盖两条文本 prompt、两种文本 API 和一条真实图文请求，共 20 条。temperature=0、top_p=1、seed=0，其余采样参数显式固定。

本轮显式使用 `--no-enable-prefix-caching`，新建基线，不执行手动清缓存。旧的 prefix cache 开启结果保留。Checkpoint 与 Level 2 共用三轮恢复证据，不代表独立的 awake 热重载测试。

| 用例 | 结果 | 生命周期 | 完成循环 |
| --- | --- | --- | --- |
| sleep_level2_determinism_prefix_off | failed | passed | 3 |
| checkpoint_reload_per_dp_prefix_off | failed | passed | 3 |
| sleep_level1_determinism_prefix_off | failed | failed | 0 |

## 严格一致性对照

完全一致要求文本、token 序列和逐 token logprob 全部精确相等，不设置容差。token 不一致时不计算对齐 logprob 差。

| 用例 | 对照 | 完全一致 | 文本一致 | token 一致 | 对齐 logprob 最大差 |
| --- | --- | --- | --- | --- | --- |
| sleep_level2_determinism_prefix_off | unchanged-weights-control-0-vs-1 | 16/20 | 16/20 | 16/20 | 0 |
| sleep_level2_determinism_prefix_off | unchanged-weights-control-0-vs-2 | 16/20 | 16/20 | 16/20 | 0 |
| sleep_level2_determinism_prefix_off | cycle1-before-vs-after | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | cycle1-after-vs-repeat | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | initial-baseline-vs-cycle1-after | 16/20 | 16/20 | 16/20 | 0 |
| sleep_level2_determinism_prefix_off | warm-baseline-vs-cycle1-repeat | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | cycle2-before-vs-after | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | cycle2-after-vs-repeat | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | initial-baseline-vs-cycle2-after | 16/20 | 16/20 | 16/20 | 0 |
| sleep_level2_determinism_prefix_off | warm-baseline-vs-cycle2-repeat | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | cycle3-before-vs-after | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | cycle3-after-vs-repeat | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level2_determinism_prefix_off | initial-baseline-vs-cycle3-after | 16/20 | 16/20 | 16/20 | 0 |
| sleep_level2_determinism_prefix_off | warm-baseline-vs-cycle3-repeat | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | unchanged-weights-control-0-vs-1 | 16/20 | 16/20 | 16/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | unchanged-weights-control-0-vs-2 | 16/20 | 16/20 | 16/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | cycle1-before-vs-after | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | cycle1-after-vs-repeat | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | initial-baseline-vs-cycle1-after | 16/20 | 16/20 | 16/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | warm-baseline-vs-cycle1-repeat | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | cycle2-before-vs-after | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | cycle2-after-vs-repeat | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | initial-baseline-vs-cycle2-after | 16/20 | 16/20 | 16/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | warm-baseline-vs-cycle2-repeat | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | cycle3-before-vs-after | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | cycle3-after-vs-repeat | 20/20 | 20/20 | 20/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | initial-baseline-vs-cycle3-after | 16/20 | 16/20 | 16/20 | 0 |
| checkpoint_reload_per_dp_prefix_off | warm-baseline-vs-cycle3-repeat | 20/20 | 20/20 | 20/20 | 0 |
| sleep_level1_determinism_prefix_off | unchanged-weights-control-0-vs-1 | 16/20 | 16/20 | 16/20 | 0 |
| sleep_level1_determinism_prefix_off | unchanged-weights-control-0-vs-2 | 16/20 | 16/20 | 16/20 | 0 |

## 生命周期与证据

两种 level 的两端运行配置证据位于 `logs/*_prefix_off-runtime-config.json`。每次 sleep/wake/reload 的 HTTP 结果及原始请求响应保存于对应 JSON 的 steps/captures。HBM 释放量和恢复后的占用见 memory_checks。完整服务日志保存于 logs。Checkpoint 的 worker 加载日志只是加载路径证据，未做逐 tensor 哈希审计。

| 用例 | 三轮恢复前后完全一致 | 恢复前后一致性状态 |
| --- | --- | --- |
| sleep_level2_determinism_prefix_off | 60/60 | passed |
| checkpoint_reload_per_dp_prefix_off | 60/60 | passed |
| sleep_level1_determinism_prefix_off | 0/0 | failed |

### sleep_level1_determinism_prefix_off 执行错误

```text
Traceback (most recent call last):
  File "/path/to/local-artifact", line 64, in main
    step('sleep',prefix+'-sleep')
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/path/to/local-artifact", line 57, in step
    if response['returncode']:raise RuntimeError('Lifecycle/probe failed: '+label)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Lifecycle/probe failed: cycle1-sleep

```

旧结果详见 开启缓存时的逐 DP 历史报告（未收录）。本轮结果只适用于关闭 prefix cache 的配置，不能据此宣称开启缓存的路径已经修复。

## 验收口径修正

Level 2 和 Checkpoint 的三轮恢复前后一致性、生命周期及内存检查通过，RL 恢复用例判为 passed。上表的旧 failed 指首次请求也计入的整体严格确定性，单独保留为 strict_determinism_status；图文差异在 RL 操作之前已复现，不算 RL 引入。Level 1 仍在锁页 CPU 权重备份阶段报 aclrtMallocHostWithCfg / 207001，未完成唤醒。两机服务已停止，16 卡空闲，临时路由已回滚。
