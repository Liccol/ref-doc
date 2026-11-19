# PR 状态与本地基线核对（2026-11-17）

查询主仓 GitHub Pull Requests API；只对本表 25 项负责，不代表穷尽所有关联 PR。merged_at 有值才算已合入。merge SHA 缺失于本地对象库时记录 object_missing，不推断为已执行祖先检查且不在基线。

本地 vLLM：9986df412141231516b80c023a799136522c3d47；Ascend：b53996aad5d959f0b4511b4b87aadb2eb425b045。

| PR | 状态 | merged_at（UTC） | 本地 vLLM 祖先核对 | merge SHA |
| --- | --- | --- | --- | --- |
| [#62407](https://github.com/vllm-project/vllm/pull/62407) | merged | 2026-10-28 21:04:17 | yes | e10ef091baf01b3ba17b93e49e0a94228c36cbfb |
| [#64336](https://github.com/vllm-project/vllm/pull/64336) | merged | 2026-10-31 06:50:27 | yes | b5755c3ba5d4f751445be39c8ad8a50d9a8f93c1 |
| [#64692](https://github.com/vllm-project/vllm/pull/64692) | merged | 2026-11-02 19:54:09 | yes | 26d198bd56ae974201fd56e5150acb8b1e29042b |
| [#65281](https://github.com/vllm-project/vllm/pull/65281) | merged | 2026-11-03 09:19:57 | yes | 66360a5cd0ad1158b2c950f924671fca68a948ff |
| [#64357](https://github.com/vllm-project/vllm/pull/64357) | merged | 2026-10-30 22:53:16 | yes | 14a5bb4adcc86e4c953236e048bae28512c92427 |
| [#57678](https://github.com/vllm-project/vllm/pull/57678) | merged | 2026-11-02 13:53:39 | yes | f9db1756e9e6809b209bb906c950ac577fdef90f |
| [#65914](https://github.com/vllm-project/vllm/pull/65914) | merged | 2026-11-06 05:59:19 | yes | 51ae6f4da19769fb3a14152a09590ebf357dcdea |
| [#64886](https://github.com/vllm-project/vllm/pull/64886) | merged | 2026-11-05 06:09:06 | yes | e5d0ef3527c90ff14707b0b209d113e573615aab |
| [#64352](https://github.com/vllm-project/vllm/pull/64352) | merged | 2026-11-06 06:24:10 | yes | ba15685ab7f4f5fbed8630bbb7441f8837e76fbe |
| [#64691](https://github.com/vllm-project/vllm/pull/64691) | merged | 2026-10-30 22:41:05 | yes | be712f5601dc7eebf43aa42848be6a84256dcada |
| [#66010](https://github.com/vllm-project/vllm/pull/66010) | merged | 2026-11-06 20:34:00 | yes | f8b116409c4cdf02cee4bb7eca8e6bb9780c2e35 |
| [#65296](https://github.com/vllm-project/vllm/pull/65296) | merged | 2026-11-03 09:18:13 | yes | 22d05b87bbc98be5974922fa372076bd6fd579af |
| [#65347](https://github.com/vllm-project/vllm/pull/65347) | merged | 2026-11-16 08:34:17 | yes | 0cf8b0f8bab1ed6465401949bdb82d87179ea8d0 |
| [#65487](https://github.com/vllm-project/vllm/pull/65487) | merged | 2026-11-16 07:05:03 | yes | d5bc1e4d53730ad7ee26f608e4be4fda1d593d58 |
| [#65802](https://github.com/vllm-project/vllm/pull/65802) | merged | 2026-11-16 13:53:23 | object_missing | a4910b41280ae228bcc74ebc9d21144e09ef63ef |
| [#66118](https://github.com/vllm-project/vllm/pull/66118) | merged | 2026-11-16 13:18:41 | object_missing | 902364d152ff5cc204addd049c217e3bd68cc2fe |
| [#65353](https://github.com/vllm-project/vllm/pull/65353) | open | — | 未检查（未合入） | b78888bcb3432525d85766cb9f25ea6565d41279 |
| [#65799](https://github.com/vllm-project/vllm/pull/65799) | open | — | 未检查（未合入） | 2c44d19d83dba1d1809a8482dd769fe945cf48ce |
| [#66002](https://github.com/vllm-project/vllm/pull/66002) | open | — | 未检查（未合入） |  |
| [#66487](https://github.com/vllm-project/vllm/pull/66487) | open | — | 未检查（未合入） | a0379c01b0b58f74d255690a6c7b657070e5e8b7 |
| [#66666](https://github.com/vllm-project/vllm/pull/66666) | draft | — | 未检查（未合入） |  |
| [#68043](https://github.com/vllm-project/vllm/pull/68043) | draft | — | 未检查（未合入） | 42844ee9325c99cfa0a8fd2c2cf6a1eea9038a22 |
| [#68032](https://github.com/vllm-project/vllm/pull/68032) | draft | — | 未检查（未合入） | f1ce586c1d91402455ec4388da1ff9cc9c1cf5e2 |
| [#65562](https://github.com/vllm-project/vllm/pull/65562) | draft | — | 未检查（未合入） |  |
| [#60401](https://github.com/vllm-project/vllm/pull/60401) | draft | — | 未检查（未合入） |  |
