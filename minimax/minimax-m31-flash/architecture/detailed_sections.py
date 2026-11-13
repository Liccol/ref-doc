"""Reading-order and representative decoder plates for the detailed diagram."""


def build_sections(Plate, colors):
    p = Plate("01-overview", 3480, 830, "01 先看执行顺序：视觉先编码 → 两类 token 合流 → 前 3 层 → 后 57 层 → 输出", "blue")
    p.box(40, 110, 700, ["语言 token IDs", "input_ids [B,S] / 打包 [N]", "整数 ID，含视觉占位位置"], 137)
    p.box(860, 110, 700, ["文本 Embedding", "W_E [200064,6144]", "→ 文本位置的向量 [N_text,6144]"], 137)
    p.box(1770, 110, 700, ["图像 / 视频", "pixels [Nv,1176] + grid_thw", "先经视觉塔；不直接输入语言层"], 137, colors["teal"][1])
    p.box(2630, 110, 810, ["独立视觉塔（完整展开见 06）", "Patch Embed → ViT ×32 → Projector → Merger", "→ 视觉 token 向量 [N_visual,6144]"], 137, colors["teal"][1])
    p.arrow([(740, 178), (860, 178)]); p.arrow([(2470, 178), (2630, 178)])
    p.box(1040, 350, 1400, ["按占位位置合并，形成同一条序列 X⁰ [N,6144]", "示意：[语言向量] [视觉向量] [视觉向量] [语言向量]", "N=N_text+N_visual；边界符等非视觉位置计入 N_text"], 137)
    p.arrow([(1210, 247), (1210, 350)])
    p.arrow([(3035, 247), (3035, 418), (2440, 418)])
    p.box(40, 575, 960, ["前 3 层：0、1、2（单独展开见 02）", "MSA + Dense FFN ×3", "两类 token 都经过相同的层内流程", "[N,6144] → [N,6144]"], 177, colors["purple"][1])
    p.box(1160, 575, 1080, ["后 57 层：3 … 59（单独展开见 03）", "MSA + MoE ×57", "每 token：Top-4 路由专家 + 1 共享专家", "[N,6144] → [N,6144]"], 177, colors["green"][1])
    p.box(2390, 575, 1050, ["Final Norm → LM Head → 采样", "选定位置 [R,6144] → logits [R,200064]", "采样新语言 token ID → 下一轮 Embedding", "普通生成 R=B；全位置输出 R=N"], 177)
    p.arrow([(1740, 487), (1740, 530), (520, 530), (520, 575)])
    p.arrow([(1000, 663), (1160, 663)]); p.arrow([(2240, 663), (2390, 663)])
    p.text(40, 800, "层内权重和 KV 内容各层独立；同构层只展开一次。positions / 长度 / slot_mapping / block_table 随请求传入，见 04。", "small")
    overview = p

    def layer_prefix(p, center, left, width, family):
        p.box(left, 110, width, ["同一序列：语言 token + 视觉 token 的向量", "Xℓ [N,6144]；进入层后不按模态分两条专家路"], 102)
        p.down(center, 212, 250)
        p.box(left, 250, width, ["Norm₁：GemmaRMSNorm [N,6144]", ("W [6144]；ε=10⁻⁶；增益 1+W", "weight")], 102)
        p.down(center, 352, 390)
        p.box(left, 390, width, ["MSA：token 间的信息交互（见 04、05）", "输入 [N,6144] → A [N,6144]"], 102, colors["orange"][1])
        p.down(center, 492, 530); p.add(center, 550)
        p.arrow([(left, 161), (35, 161), (35, 550), (center-20, 550)])
        p.text(48, 520, "原 Xℓ 残差", "small")
        p.down(center, 570, 610)
        p.box(left, 610, width, ["U=Xℓ+A；Z=Norm₂(U) [N,6144]", ("第二套独立 Norm 权重 [6144]；Z 进入 "+family, "weight")], 102)

    p = Plate("02-dense-layer", 1420, 2220, "02 前 3 层：MSA + Dense FFN，展开一层", "purple")
    layer_prefix(p, 710, 330, 760, "Dense FFN")
    p.box(330, 765, 760, ["Dense FFN 输入 Z [N,6144]"], 68, colors["green"][1])
    p.down(710, 712, 765)
    p.arrow([(710, 833), (710, 863), (345, 863), (345, 900)])
    p.arrow([(710, 863), (1075, 863), (1075, 900)])
    p.box(95, 900, 500, ["Gate Linear", "G [N,12288]", ("Wgate [12288,6144]", "weight")], 137, colors["green"][1])
    p.box(825, 900, 500, ["Up Linear", "V [N,12288]", ("Wup [12288,6144]", "weight")], 137, colors["green"][1])
    p.box(330, 1110, 760, ["SwiGLU-OAI：逐通道门控相乘", "G′=min(G,7)；V′=clamp(V,−7,7)", "G′×sigmoid(1.702G′)×(V′+1)", "输出 [N,12288]"], 177, colors["green"][1])
    p.arrow([(345, 1037), (345, 1070), (530, 1070), (530, 1110)])
    p.arrow([(1075, 1037), (1075, 1070), (890, 1070), (890, 1110)])
    p.down(710, 1287, 1330)
    p.box(330, 1330, 760, ["Down Linear → F [N,6144]", ("Wdown [6144,12288]", "weight")], 102, colors["green"][1])
    p.box(95, 1510, 500, ["所有有效 token 走同一套 FFN", "语言/视觉位置都不跳过", "不做专家选择，没有 router"], 137)
    p.box(825, 1510, 500, ["本层只做逐 token 通道加工", "不同位置用同一套本层权重", "不同层的权重仍彼此独立"], 137)
    p.arrow([(710, 1432), (710, 1770)]); p.add(710, 1790)
    p.arrow([(1090, 645), (1370, 645), (1370, 1790), (730, 1790)])
    p.text(1150, 1750, "U 残差", "small")
    p.down(710, 1810, 1850)
    p.box(330, 1850, 760, ["Xℓ⁺¹=U+F [N,6144]", "依次完成层 0、1、2 → 进入 MoE 层 3"], 102)
    p.box(95, 1995, 1230, ["Dense 的含义：FFN 不做专家稀疏路由", "它的 attention 仍然是 MSA，不是全量 dense attention。", "FFN 中间宽度 I=12288；输出宽度回到 6144，可与 U 相加。"], 137)
    dense = p

    p = Plate("03-moe-layer", 2000, 2220, "03 后 57 层：MSA + MoE，展开一层的两条专家路", "green")
    layer_prefix(p, 1000, 500, 1000, "MoE")
    p.down(1000, 712, 765)
    p.box(650, 765, 700, ["同一份 Z [N,6144] 同时送两条分支"], 68)
    p.arrow([(1000, 833), (1000, 860), (590, 860), (590, 900)])
    p.arrow([(1000, 860), (1605, 860), (1605, 900)])
    p.box(75, 900, 1030, ["路由分支：Router Linear → sigmoid", "Wrouter [128,6144]（FP32）→ scores s [N,128]", "按 s + correction_bias [128] 为每 token 选 Top-4"], 137)
    p.box(1295, 900, 630, ["共享分支：1 个 Shared Expert", "不经过 router，不占 Top-4 名额", "每一个 token 都进入此分支"], 137, colors["teal"][1])
    p.down(590, 1037, 1075)
    p.box(75, 1075, 1030, ["IDs [N,4]；取原始 s 并归一化 → r [N,4]", "Dispatch：将 token 按选中的 expert 分组", "Σnₑ=4N；每个 expert e 收到 Zₑ [nₑ,6144]"], 137)
    p.down(590, 1212, 1250)
    p.box(75, 1250, 1030, ["128 个独立路由专家中，每 token 只计算选中的 4 个", "每个专家：6144 → gate/up 各 3072 → 激活 → down 6144", ("W1/W3 [3072,6144]；W2 [6144,3072]", "weight"), "专家输出 Yₑ [nₑ,6144]，保留原 token ID 用于回收"], 177)
    p.arrow([(1610, 1037), (1610, 1250)])
    p.box(1295, 1250, 630, ["Shared FFN", "[N,6144] → gate/up 各 [N,3072]", "→ SwiGLU-OAI → down", "Shared(Z) [N,6144]"], 177, colors["teal"][1])
    p.down(590, 1427, 1465)
    p.box(75, 1465, 1030, ["Combine：按原 token ID 回收四个专家结果", "Rₙ=2.0×Σₖ rₙₖ·Yₙₖ → R [N,6144]", "权重只用于路由分支；共享输出不乘 routed factor"], 137)
    p.add(1000, 1660)
    p.arrow([(590, 1602), (590, 1660), (980, 1660)])
    p.arrow([(1610, 1427), (1610, 1660), (1020, 1660)])
    p.down(1000, 1680, 1710)
    p.box(650, 1710, 700, ["F=R+Shared(Z) [N,6144]"], 68)
    p.down(1000, 1778, 1790); p.add(1000, 1810)
    p.arrow([(1500, 645), (1960, 645), (1960, 1810), (1020, 1810)])
    p.text(1760, 1775, "U 残差", "small")
    p.down(1000, 1830, 1860)
    p.box(500, 1860, 1000, ["Xℓ⁺¹=U+F [N,6144]", "重复至层 59 完成 → Final Norm → LM Head"], 102)
    p.box(75, 2000, 1850, ["不是“语言 token 走路由、视觉 token 走共享”：两类 token 都同时进入 router 和共享分支。", "路由选择取决于该 token 当时的 hidden state；不同 token 可以选择不同的 4 个专家。", "与 Dense 相比：变化在 FFN 内部；层输入、MSA、两次 Norm/残差及输出 shape 保持同类结构。"], 137)
    moe = p

    p = Plate("05-msa-features-and-ffn", 2300, 1940, "05 MSA 的特征，以及 Dense FFN / MoE 到底差在哪", "orange")
    p.box(50, 110, 2200, ["MSA = 先用独立 index Q/K 找重要历史块，再用主 Q/K/V 算注意力", "所有 60 层都使用；block_size=128；每 index head 选 Top-16 块，强制包含当前 local 块。", "GQA：64 个 Q heads 分 4 组，每组 16 个 Q heads，共用对应 KV head 与选块结果。", "稀疏的是主 attention 的历史读取范围；完整历史 K/V 通常仍保留，未来轮次可重新选中。"], 177)
    p.text(50, 340, "选块示意：24 个历史块中每组选 16 个；23 为当前 local；24/25 表示未来，不可访问", "node")
    for group in range(4):
        y=390+group*105
        p.text(50, y+31, f"KV 组 {group}", "body")
        selected=set(range(group*2,group*2+15))|{23}
        assert len(selected)==16
        for block in range(26):
            x=210+block*78
            fill="#ffe3b4" if block in selected else "#edf0f4"
            if block==23:fill="#f5c7bb"
            p.rect(x,y,64,52,fill)
            p.text(x+32,y+34,f"{block:02}","small","middle")
            if block>=24:
                p.items.append(f'<line x1="{x+5}" y1="{y+45}" x2="{x+59}" y2="{y+7}" stroke="#617186" stroke-width="2"/>')
    p.text(50, 850, "橙：index 分数选中　红：强制 local（占 16 块预算）　灰：未选历史　斜线：未来被因果掩码排除", "small")
    p.box(150, 900, 2000, ["每组最多 16×128 个候选 key 位置 → 主 QKᵀ/√128 → causal mask → softmax → V", "4 组可以选不同的块；同组 16 个 Q 头用不同 Q，所以注意力概率和输出仍可以不同。", "index 分数用于选块，不直接作为主 attention 概率；末块不满、短上下文会减少有效位置。"], 137)
    p.text(50, 1110, "两种 FFN 对比：输入和输出均为 [N,6144]，区别在“每个 token 使用哪些参数”", "node")
    p.box(50, 1150, 1060, ["Dense FFN：前 3 层", "没有 router，也没有 expert dispatch", "所有 token 使用本层同一套 FFN 权重", "gate/up 中间宽度 I=12288", "6144 → 12288 → 6144", "每 token 完整计算这一套 FFN"], 247, colors["purple"][1])
    p.box(1190, 1150, 1060, ["MoE FFN：后 57 层", "router 选专家 → dispatch → combine", "128 个路由专家各有独立权重", "每 token 激活 Top-4 + 1 个共享专家", "每个专家：6144 → 3072 → 6144", "加权路由结果 + 无条件共享结果"], 247, colors["green"][1])
    p.box(50, 1460, 2200, ["Attention 与 FFN 是两个不同维度的设计", "MSA 决定“这个 token 从哪些历史 token 获取信息”；Dense/MoE 决定“这个 token 由哪套 FFN 参数加工”。", "所以前 3 层是 MSA + Dense FFN，后 57 层是 MSA + MoE；Dense FFN 不意味着全量 attention。"], 137)
    p.box(50, 1660, 2200, ["为什么保留共享专家？", "结构上它为所有 token 提供一条无条件计算路径，与按 token 选择的专家输出相加。", "不能仅凭结构断言某个专家专管语言或视觉，也不能直接推断 MoE 必然比 Dense 更快。", "分发、通信、量化、批量与负载都会影响实际性能；此图只解释逻辑计算和数据流。"], 177, "#f3f6fa")
    return overview, dense, moe, p
