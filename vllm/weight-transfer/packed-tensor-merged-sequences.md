# Packed tensor 四方法合并时序对照

文档版本：**v1.0.26** · 更新日期：2026-11-13

对照 vLLM main `24420b7a` 与 Ascend 重构前基线 `37854b83`。共同步骤只画一次，`alt` 分支和注释标注两侧差异；分支是文档对照，不是运行时选择，也不是两套实现互相调用。当前继承 PR 的适配说明见 [主设计文档](packed-tensor-design.md)，原有分图继续保留。

## 1. HCCL producer

```mermaid
sequenceDiagram
    participant L as producer
    participant S as slot streams
    participant F as post_iter_func
    participant G as communicator
    alt vLLM
        L->>S: _get_streams：缓存 CUDA streams
    else Ascend
        L->>L: 校验容量与 slot 数
        L->>S: 新建 NPU streams
    end
    loop 每个有 payload 的 chunk
        L->>S: synchronize 当前 slot，进入 stream 上下文
        Note over L,S: vLLM 在同步后、进入上下文前清除该 slot 的 in_flight
        loop 取 tensor，累计字节超过阈值或到 EOF
            L->>F: callback(next(iterator))
            F-->>L: tensor
            Note over L,F: Ascend 额外校验 Tensor / dtype / numel
            L->>L: contiguous，展平为 uint8，累加字节
        end
        L->>L: cat 字节 slices
        Note over L: vLLM 由 pack_tensors 返回 PackedChunk；Ascend 内联打包
        alt vLLM
            Note over L,G: 本循环未显式等待其他 slot 的前次通信
            L->>G: broadcast(packed, src)
            L->>L: 保存 in_flight，轮转 slot，退出上下文
        else Ascend
            opt 前次广播在其他 slot
                L->>S: synchronize previous_broadcast_stream
            end
            L->>G: broadcast(packed, src, stream=当前 slot)
            L->>L: 保存前次广播 stream，退出上下文，轮转 slot
        end
    end
    alt vLLM
        L->>S: 正常结束 drain；异常可提前跳出
    else Ascend
        L->>S: finally drain；失败时保留 buffer 并处理异常优先级
    end
```

等待当前 slot 不等于等待另一个 slot 的前次广播。当前 PR 用 `_HCCLPackedGroup` 在 native HCCL 前后同步来承担通信边界，不是简单删除旧版的跨 stream 保序。上图展示旧实现，不包含这个修改后的适配器。

## 2. HCCL consumer

```mermaid
sequenceDiagram
    participant L as consumer
    participant S as slot streams / events
    participant M as metadata iterator
    participant G as communicator
    participant F as post_unpack_func
    alt vLLM
        L->>S: 获取缓存 streams，保存 caller_stream
    else Ascend
        L->>L: 校验容量与 slot 数
        L->>S: 新建 NPU streams
    end
    loop 每个有 metadata 的 chunk
        alt vLLM main
            L->>S: 等待当前 slot_done event，再进入 slot stream
        else Ascend
            L->>S: synchronize 当前 slot，再进入 slot stream
        end
        L->>M: 收集 name / shape / dtype，计算字节数
        Note over L,M: Ascend 额外检查负 shape
        L->>L: 达到分包边界或 EOF，分配 uint8 接收 buffer
        Note over L: vLLM 用 device 参数；Ascend 固定 NPU 分配
        alt vLLM
            L->>G: 在 slot 上下文内 broadcast(packed, src)
            L->>S: wait/record prev_received
            L->>L: packed.record_stream(全部 slot 和 caller)
            L->>L: split / contiguous / view(dtype) / view(shape)
            L->>F: 传入共享 packed storage 的 views
            L->>S: record slot_done，退出上下文
        else Ascend
            L->>S: 退出 metadata/分配上下文
            L->>G: broadcast(packed, src, stream=当前 slot)
            L->>S: synchronize 当前 slot，再进入 slot 上下文
            L->>L: split / clone 字节 / view(dtype) / reshape
            L->>F: 传入独立 storage tensors
            L->>S: 退出上下文
        end
        L->>L: 轮转 slot
    end
    alt vLLM
        L->>S: 正常结束 drain 全部 slot
    else Ascend
        L->>S: finally drain，失败时保留 buffer
    end
```

上游的 event 和 `record_stream` 保护 raw packed buffer。views 与独立 clone 的区别不等同于“乱序/有序”；独立 clone 仍需要正确的读依赖和 reader 生命周期保护。图示 main 的 slot_done 也不能泛化为所有 tag 的实现。

## 3. IPC producer

```mermaid
sequenceDiagram
    participant C as trainer / 调用方
    participant L as IPC producer
    participant B as 固定容量共享 buffer
    participant F as post_iter_func
    participant S as 当前 stream
    C->>L: next(generator)
    alt vLLM
        L->>B: empty(capacity, uint8, device=cuda)
    else Ascend
        L->>L: 校验 capacity 为正
        L->>B: empty(capacity, uint8, device=npu)
    end
    L->>B: reduce_tensor：一次导出，保存 handle args
    loop 每个 tensor
        L->>F: callback(name, source)
        F-->>L: tensor
        Note over L,F: Ascend 额外校验 Tensor / dtype / numel
        L->>L: contiguous，展平为 uint8
        Note over L: 两侧单 tensor 超容量都报 ValueError
        opt 已有非零字节且加入本 tensor 将超过容量
            L->>S: synchronize：完成当前 chunk 的 copy
            alt vLLM
                L-->>C: yield PackedIpcChunk(metadata, handle)
            else Ascend
                L-->>C: yield dict(metadata, handle)
            end
            Note over C,B: 调用方保证消费完成后，才能复用共享 buffer
            C->>L: next：恢复生成器
            L->>L: 换新 metadata lists，清零字节计数
        end
        L->>B: copy 到累计字节对应区间
        L->>L: 追加 metadata，累加字节
    end
    alt vLLM
        opt EOF 且 total_bytes 非零
            L->>S: synchronize
            L-->>C: yield 最后一个 PackedIpcChunk
        end
    else Ascend
        opt EOF 且 names 非空
            L->>S: synchronize
            L-->>C: yield 最后一个 dict，包含全零元素 metadata
        end
    end
    Note over L,C: 完全空输入两侧均不 yield；返回类型是本地可适配差异
```

“全零元素”指 `numel() == 0`，不是值全为零。`yield` 交付 metadata 和共享 handle 后暂停，不复制整个共享 buffer。返回类型、设备分配和 EOF 发送条件是三个独立差异。

## 4. IPC consumer

```mermaid
sequenceDiagram
    participant C as engine / 调用方
    participant L as IPC consumer
    participant D as 设备标识来源
    participant R as importer / reduction
    participant B as 共享 buffer
    C->>L: handle / metadata / device_index
    alt vLLM
        L->>D: CUDA get_device_properties(device_index).uuid
        D-->>L: physical_gpu_id
    else Ascend
        Note over C,L: 调用方额外传 physical_npu_id
        L->>L: 校验 device_index
    end
    L->>L: 根据物理标识查 handle，不存在则报错
    opt Ascend 特有的重建前校验
        L->>L: 校验 dtype、四表长度、shape/size 和每项字节数
        L->>L: 校验 handle args 含设备索引
    end
    L->>L: 复制 args，覆写接收端 device_index
    alt vLLM
        L->>R: importer.rebuild(args)，未传时创建临时 importer
        R->>B: 缓存命中复用映射；新 export 用 rebuild_cuda_tensor
        Note over C,R: 跨 chunk 需共享 importer，由 owner 负责 close
    else Ascend
        L->>R: 每次调用 rebuild_npu_tensor(args)
        R->>B: 每次重建映射，无 importer cache
    end
    B-->>L: packed 共享映射
    Note over L,B: Ascend 额外检查有效字节数不超过 buffer 容量
    L->>B: 截取 sum(tensor_sizes) 有效字节
    alt vLLM
        L->>L: 解析 dtype；split / contiguous / view(dtype) / view(shape)
        L->>L: 再逐项 clone
    else Ascend
        L->>L: 按 metadata 取字节切片
        L->>L: 先 clone 字节，再 view(dtype) / reshape
    end
    L-->>C: 返回独立 storage tensors
```

设备标识、重建缓存和解包顺序分别标注。本图描述修改前现状，不代表主设计文档第 6 章的硬件扩展点已经落地；本次只补图，不修改生产代码或验证结论。
