"""Generate detailed, editable architecture plates from the verified config.

Run: python draw_detailed.py --png
SVGs are editable sources. PNG export uses installed Chrome/Edge and Pillow.
"""

from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "detailed"
OUT.mkdir(exist_ok=True)
CONFIG = json.loads((ROOT / "sources/config.json").read_text(encoding="utf-8"))
T = CONFIG["text_config"]
assert (T["hidden_size"], T["num_hidden_layers"], T["head_dim"]) == (6144, 60, 128)
assert sum(T["moe_layer_freq"]) == 57

COLORS = {"purple": ("#7758a6", "#f3eef9"), "orange": ("#b9741d", "#fff5e5"),
          "green": ("#4b853e", "#f0f7ec"), "teal": ("#237e76", "#eaf7f4"),
          "red": ("#ad5a53", "#fff0ee"), "blue": ("#447da5", "#eef5fb")}
STYLE = """<style>
text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#22364c}
.title{font-size:44px;font-weight:700}.head{font-size:31px;font-weight:700}
.node{font-size:25px;font-weight:600}.body{font-size:24px}.small{font-size:22px}
.weight{font-size:22px;fill:#59677a}.note{font-size:22px;fill:#59677a}
</style><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0 0L10 5L0 10Z" fill="#617186"/></marker></defs>"""


class Plate:
    def __init__(self, name, w, h, title, color):
        self.name, self.w, self.h = name, w, h
        self.stroke, self.fill = COLORS[color]
        self.items = []
        self.text_bounds = []
        self.rect(1, 1, w - 2, h - 2, "#fbfcfe", self.stroke, 18)
        self.text(28, 49, title, "head")
        self.items.append(f'<line x1="28" y1="75" x2="{w-28}" y2="75" stroke="{self.stroke}" opacity=".35"/>')

    def rect(self, x, y, w, h, fill, stroke="#c9d4e0", radius=10):
        assert 0 <= x and 0 <= y and x + w <= self.w and y + h <= self.h
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')

    def text(self, x, y, text, cls="body", anchor="start"):
        assert 0 <= x <= self.w and 0 <= y <= self.h
        self.items.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>')
        self.text_bounds.append((x, y, text, cls, anchor))

    def box(self, x, y, w, lines, h=None, fill=None):
        h = h or (35 * len(lines) + 28)
        self.rect(x, y, w, h, fill or self.fill)
        for i, line in enumerate(lines):
            if isinstance(line, tuple):
                label, cls = line
            else:
                label, cls = line, "node" if i == 0 else "body"
            self.text(x + w / 2, y + 35 + i * 35, label, cls, "middle")
            self.items[-1] = self.items[-1].replace('<text ', f'<text data-max-width="{w-28}" ')
        return y + h

    def arrow(self, points, dashed=False):
        for x, y in points:
            assert 0 <= x <= self.w and 0 <= y <= self.h
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.items.append(f'<polyline points="{coords}" fill="none" stroke="#617186" stroke-width="2.2"{dash} marker-end="url(#arrow)"/>')

    def down(self, x, y1, y2):
        self.arrow([(x, y1), (x, y2)])

    def add(self, x, y):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="20" fill="white" stroke="#617186" stroke-width="2"/>')
        self.text(x, y + 8, "+", "node", "middle")

    def render(self):
        body = "\n".join(self.items)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img"><title>{escape(self.name)}</title>'
                f'<rect width="{self.w}" height="{self.h}" fill="white"/>{STYLE}{body}</svg>')


# MSA plate: explicit independent projections and cached/indexed paths.
p = Plate("04-msa", 1120, 1940, "04 MSA 内部：独立索引找块，再算主注意力", "orange")
p.box(235, 105, 650, ["归一化输入 H [N,6144]", ("来自 02 / 03 的 Norm₁；positions [N]", "weight")], 102)
p.arrow([(560, 207), (560, 233), (285, 233), (285, 260)])
p.arrow([(560, 233), (835, 233), (835, 260)])
p.box(35, 260, 500, ["主 Q / K / V 三个 Linear", "Q [N,8192]；K/V 各 [N,512]", ("Wq [8192,6144]", "weight"), ("Wk / Wv 各 [512,6144]", "weight")], 177)
p.box(585, 260, 500, ["index Q / K 两个 Linear", "iQ [N,512]；iK [N,128]", ("WiQ [512,6144]", "weight"), ("WiK [128,6144]", "weight")], 177)
p.down(285, 437, 475); p.down(835, 437, 475)
p.box(35, 475, 500, ["reshape：只重排，不改数值", "Q [N,64,128]", "K / V 各 [N,4,128]"], 137)
p.box(585, 475, 500, ["reshape：只重排，不改数值", "iQ [N,4,128]", "iK [N,1,128]"], 137)
p.down(285, 612, 650); p.down(835, 612, 650)
p.box(35, 650, 500, ["Q/K per-head GemmaRMSNorm", "每头 128 维；W 各 [128]", "partial RoPE：前 64 维旋转", ("后 64 维不变；V 不做 Norm/RoPE", "weight")], 177)
p.box(585, 650, 500, ["iQ/iK per-head GemmaRMSNorm", "每头 128 维；W 各 [128]", "partial RoPE：前 64 维旋转", ("θ=10⁷；iK 在 4 个 iQ 头间共享", "weight")], 177)
p.down(285, 827, 865); p.down(835, 827, 865)
p.box(35, 865, 500, ["K/V 按 slot_mapping 写缓存", "每请求 K/V 各 [Tᵢ,4,128]", "旧缓存 + 本轮 K/V → 可读取历史"], 137, COLORS["red"][1])
p.box(585, 865, 500, ["index K 按槽位写缓存", "每请求 iK [Tᵢ,1,128]", "iQ 只用于本轮查询，不保留历史"], 137, COLORS["red"][1])
p.down(835, 1002, 1040)
p.box(585, 1040, 500, ["index 点积 → 按块 max", "I[n,h,m] = maxⱼ ⟨iQₙₕ,iKⱼ⟩", "block=128；Mᵢ=ceil(Tᵢ/128)", "每个查询、每个 iQ 头分别评分"], 177)
p.down(835, 1217, 1255)
p.box(585, 1255, 500, ["因果边界 + 强制 local + Top-K", "init=0；local=1；Top-K blocks=16", "块 ID：概念容量 [N,4,16]", ("local 占用 Top-K 预算；无效槽屏蔽", "weight")], 177)
p.box(35, 1070, 500, ["本轮主 Q 直接进入 attention", "64 Q heads : 4 KV heads", "组 g=floor(q_head/16)", "每组使用对应 index head 的块集"], 177)
# Q bypasses the cache; cached K/V have their own route to sparse attention.
p.arrow([(535, 738), (550, 738), (550, 1030), (285, 1030), (285, 1070)])
p.arrow([(285, 1002), (285, 1018), (15, 1018), (15, 1570), (85, 1570)])
p.arrow([(835, 1432), (835, 1480)])
p.arrow([(285, 1247), (285, 1480)])
p.box(85, 1480, 950, ["按选中块读取主 K/V → 主 sparse attention", "logits=QKᵀ/√128 → causal mask → softmax → 加权 V", "每头在有效选中 key 上归一化；输出 [N,64,128]", ("每 KV 组最多 16×128 个候选位置；不物化完整 N×T 主分数", "weight")], 177)
p.down(560, 1657, 1695)
p.box(135, 1695, 850, ["merge heads [N,8192] → O Linear → A [N,6144]", ("Wo [6144,8192]；返回 02 / 03 第一次残差加法", "weight")], 102)
p.text(40, 1840, "融合可将五路投影合成 [N,9856]，逻辑分支仍如上。", "small")
p.text(40, 1878, "这里的 index Top-K 选择块；MoE Top-K 选择专家（见 03）。", "small")
msa = p

# Entire visual path with one representative ViT layer.
p = Plate("06-vision", 3480, 1450, "06 视觉全链路：预处理 → Patch Embedding → ViT ×32 → Projector → 2×2 Merger → 占位替换", "teal")
p.box(35, 110, 1030, ["图像 / 视频预处理（dynamic resolution）", "单图教学例：处理后 672×672，复制/补齐为 2 帧", "归一化：(pixel/255−mean)/std；按邻接 merge 顺序排 patch", "grid=[1,48,48]；Nv=2304；pixel_values [2304,1176]"], 177)
p.box(1215, 110, 1030, ["Conv3d Patch Embedding：每 patch 一个向量", "reshape [Nv,3,2,14,14]；W [1280,3,2,14,14]，无 bias", "kernel=stride=(2,14,14) → [Nv,1280,1,1,1]", "flatten → [Nv,1280] → Pre LayerNorm [Nv,1280]"], 177)
p.box(2395, 110, 1050, ["ViT 全 32 层：结构相同，权重逐层独立", "层 0 → 层 1 → … → 层 31，每层 hidden=1280", "每层两次 LayerNorm + 两次 residual，单层展开如下", "取最后 encoder 输出；当前实现不应用未训练 post norm"], 177)
p.arrow([(1065, 198), (1215, 198)]); p.arrow([(2245, 198), (2395, 198)])
p.text(45, 360, "ViT 单层（对应上方 32 层中的任一层）：以下形状以不分片、所有 patch 打包为例", "node")
p.box(35, 400, 770, ["① LN₁ + QKV Linear", "输入 [Nv,1,1280]，LN gain/bias [1280]", "QKV [Nv,1,3840]", ("Wqkv [3840,1280]；bias [3840]", "weight"), "reshape Q/K/V 各 [1,Nv,16,80]"], 212)
p.box(915, 400, 770, ["② Q/K 3D partial RoPE", "时间/高/宽各 26 维，总共旋转 78 维", "head_dim=80，余下 2 维不变；V 不旋转", "cos / sin 各 [Nv,39]；θ=10000", ("Q/K 的 shape 不变，但数值随网格位置改变", "weight")], 212)
p.box(1795, 400, 770, ["③ 视觉双向 MHA + O Linear", "每头 QKᵀ/√80 → softmax → V", "[1,Nv,16,80] → [Nv,1,1280]", ("Wo [1280,1280]；bias [1280]", "weight"), "以片段边界隔离图片/视频，不用语言 MSA"], 212)
p.box(2675, 400, 770, ["④ 残差 + LN₂ + GELU MLP + 残差", "U=X+Attn(LN₁(X))", "[Nv,1,1280] → [Nv,1,5120] → 同输入宽度", ("Wfc1 [5120,1280]；Wfc2 [1280,5120]", "weight"), "Y=U+MLP(LN₂(U)) → 下一 ViT 层"], 212)
for a,b in [(805,915),(1685,1795),(2565,2675)]:p.arrow([(a,506),(b,506)])
p.text(45, 685, "视觉塔输出 → 对齐文本通道 → 压缩空间 token → 替换语言占位：顺序不可交换", "node")
p.box(35, 730, 770, ["ViT 最后输出", "V [Nv,1280]", "教学例：[2304,1280]", ("不添加 CLS；LayerNorm 与文本 RMSNorm 不同", "weight")], 177)
p.box(915, 730, 770, ["Projector：两层 Linear + GELU", "1280 → 6144 → 6144，token 数不变", ("W₁ [6144,1280]；W₂ [6144,6144]", "weight"), "[2304,1280] → [2304,6144]"], 177)
p.box(1795, 730, 770, ["2×2 空间拼接 → 两层 Merge MLP", "[Nv,6144] → [Nv/4,24576]（只重排）", ("W₁ [6144,24576]；W₂ [6144,6144]", "weight"), "GELU；教学例最终 [576,6144]"], 177)
p.box(2675, 730, 770, ["按位置替换语言 embedding", "视觉向量 [Nv/4,6144] → 同宽占位位置", "100 个非视觉 token + 576 个图像位置", "→ [676,6144] → 语言第 0 层（01 / 02）"], 177)
for a,b in [(805,915),(1685,1795),(2565,2675)]:p.arrow([(a,818),(b,818)])
p.box(35, 960, 1650, ["图像数值流转：像素归一化会改值，patchify/reshape 只重排", "Conv / Linear / GELU / Norm / attention / residual 都参与数值计算", "2×2 拼接后 4×6144=24576；可训练 MLP 才压缩到 6144，不是求均值", "所有 projector / merger bias 均为 [6144]；ViT FFN bias 为 [5120]、[1280]"], 177, "#f3f6fa")
p.box(1795, 960, 1650, ["视频教学例：已采样/补齐 8 帧，处理后每帧 672×672", "grid=[4,48,48] → Nv=9216 → pixels [9216,1176] → [2304,6144]", "temporal patch=2 先压时间；2×2 merger 仅压空间；时间戳等控制 token 另计", "当前实现按时间 grid 每 4 组限段；长视频分段双向 attention，段间隔离"], 177, "#f3f6fa")
p.box(35, 1190, 3410, ["层级覆盖：视觉 Patch Embed ×1 → Pre LN ×1 → Vision Transformer ×32 → Projector ×1 → Merger ×1 → 文本 60 层 → Final Norm → LM Head", "图中的 Nv/4 与 576 都是视觉内容向量数量；实际 dynamic resolution 以 grid 为准，672 不代表固定输入尺寸。", "视频/图片在 prefill 时变成语言前缀；正常后续 decode 复用语言 KV cache，通常不逐 token 重跑同一份视觉输入。"], 137, "#f3f6fa")
vision = p

# Runtime and optional branch, linked to the same input/output objects.
p = Plate("07-runtime", 3480, 930, "07 逐 token 轮转、cache 地址与可选 DSpark：参数不变，激活与状态推进", "red")
p.box(35, 110, 740, ["Prefill：B=2，S=256，P=0", "IDs [2,256] → 打包 [512]", "hidden [512,6144]；每请求缓存长 256", "末位置 logits [2,200064]", "采样第 257 个 ID，尚未写入其 KV"], 212)
p.box(905, 110, 740, ["Decode 1：P=256，S=1", "输入刚采样的 ID [2,1]，position=256", "hidden [2,6144]；本轮 K/V 各 [2,4,128]", "每层缓存长度 256 → 257，逻辑块 2 → 3", "采样第 258 个 ID，下一轮再进入缓存"], 212)
p.box(1775, 110, 740, ["Decode 2：P=257，S=1", "输入第 258 个 ID，position=257", "hidden [2,6144]；Q [2,64,128]", "每层缓存长度 257 → 258；当前块仍为 2", "重复 60 层 → logits → 第 259 个 ID"], 212)
p.box(2645, 110, 800, ["新 token 的下一轮闭环", "采样 ID → Embedding → 60 层", "新 K/V 写槽 → index 选块 → attention", "旧 Q 不缓存；旧 K/V 不逐轮重算", "未选旧块保留，后续 query 可重新选中"], 212)
for a,b in [(775,905),(1645,1775),(2515,2645)]:p.arrow([(a,216),(b,216)])
p.box(35, 380, 1610, ["分页地址例：假设 cache 页也为 128 token", "block_table=[9,2,15]；逻辑位置 256 → 第 2 块、offset 0", "物理页=15；slot=15×128+0=1920；位置 257 写 slot 1921", "每层每历史 token：K 512 + V 512 + index K 128 = 1152 标量", "若三类皆 BF16：60 层共 135 KiB/token；未含页尾、scale 与工作区", ("MSA 选择块=128 与 allocator 物理页是不同概念，不保证所有后端相同", "weight")], 247)
p.box(1775, 380, 1670, ["可选 DSpark 草稿分支：不属于 target 的第 61–65 层", "target 层 [3,17,31,45,59] hidden，各 [N,6144]", "拼接 [N,30720] → fc W [6144,30720] → [N,6144]", "Draft ×5：64/4 heads ×128；Dense I=12288；sliding window=4096", "Markov rank=256；confidence W [1,6400]；block_size=7", ("候选 → target 验证 → 接受部分并推进/回退状态；7 不等于每步必接收 7 token", "weight")], 247, COLORS["blue"][1])
p.box(35, 685, 3410, ["读图约定：激活 shape 默认全局逻辑维度；Linear 权重为 [out,in]，计算 X·Wᵀ；不同层独立权重、独立 KV 内容。", "TP/EP/量化会改变本地或文件布局：如 TP=4 的五路投影本地宽度为 2560；NVFP4 的单 expert W1 逻辑 [3072,6144]，U8 打包 [3072,3072]。", "实线为本区数据流；折线路径为分支/残差；编号为跨区模块引用。本图按固定配置与本地实现静态核验，示例不是模型真实激活。", "来源：HF ce7f4df87d2e05262f5775f373dac4772f134041；architecture/sources 配置与 73 项 header 抽样；公式和来源详见 model_analyze.md。"], 177, "#f3f6fa")
runtime = p

from detailed_sections import build_sections

overview, dense, moe, features = build_sections(Plate, COLORS)
plates = [overview, dense, moe, msa, features, vision, runtime]
for p in plates:
    (OUT / f"{p.name}.svg").write_text(p.render(), encoding="utf-8")

placements = [(overview,60,170),(dense,60,1050),(moe,1540,1050),
              (msa,60,3320),(features,1240,3320),(vision,60,5310),(runtime,60,6810)]
W,H=3600,7870
full=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
      '<title>MiniMax-M3.1-Flash：整体顺序、Dense 与 MoE 层、MSA 及视觉塔</title>',
      f'<rect width="{W}" height="{H}" fill="white"/>',STYLE,
      '<text x="1800" y="68" class="title" text-anchor="middle">MiniMax-M3.1-Flash · 先看整体，再看 token 在层内怎么走</text>',
      '<text x="1800" y="116" class="body" text-anchor="middle">视觉塔先编码并合入语言序列 → 3 层 MSA + Dense FFN → 57 层 MSA + MoE → 输出；同构层只展开一次</text>']
for p,x,y in placements:
    full.append(f'<g transform="translate({x},{y})">'+"\n".join(p.items)+"</g>")
full.append('<text x="60" y="7815" class="note">2026-11-13 · 可编辑 SVG / 高清 PNG / 分区图；W 为参数，X/H/Q/K/V 为激活；参数不随普通推理轮次更新。</text></svg>')
(ROOT / "minimax-m31-detailed-architecture.svg").write_text("\n".join(full),encoding="utf-8")
(OUT / "layout.json").write_text(json.dumps({"width":W,"height":H,"plates":[{"name":p.name,"x":x,"y":y,"width":p.w,"height":p.h} for p,x,y in placements]},indent=2)+"\n",encoding="utf-8")
print(f"Generated full SVG and {len(plates)} detail plates in {ROOT}")


def render_png():
    """Render and check text widths, then export full-size and readable plates."""
    import html
    import os
    import re
    import subprocess
    import tempfile
    from PIL import Image

    choices = [Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
               Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"]
    browser = next((b for b in choices if b.exists()), None)
    if browser is None:
        raise RuntimeError("PNG export requires installed Chrome or Edge; SVGs were generated.")
    svg = (ROOT / "minimax-m31-detailed-architecture.svg").read_text(encoding="utf-8")
    script = """<script>const bad=[];
document.querySelectorAll('text[data-max-width]').forEach(t=>{
if(t.getBBox().width>+t.dataset.maxWidth)bad.push({text:t.textContent,width:t.getBBox().width,max:+t.dataset.maxWidth})});
const pre=document.createElement('pre');pre.id='qa';pre.textContent=JSON.stringify(bad);document.body.append(pre);</script>"""
    with tempfile.TemporaryDirectory(prefix="minimax-architecture-") as temp:
        work = Path(temp)
        page = work / "render.html"
        page.write_text('<!doctype html><meta charset="utf-8"><style>body{margin:0;background:white}</style>'+svg+script, encoding="utf-8")
        command = [str(browser), "--headless", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--disable-extensions",
                   "--user-data-dir="+str(work / "profile"), "--force-device-scale-factor=1.5",
                   f"--window-size={W},{H}", "--screenshot="+str(ROOT / "minimax-m31-detailed-architecture.png"),
                   "--dump-dom", page.as_uri()]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        match = re.search(r'<pre id="qa">(.*?)</pre>', result.stdout, re.S)
        assert match, "Browser text-layout QA did not complete"
        assert not json.loads(html.unescape(match.group(1))), match.group(1)
    image = Image.open(ROOT / "minimax-m31-detailed-architecture.png")
    assert image.size == (round(W*1.5), round(H*1.5)), image.size
    scale = image.width / W
    for plate, x, y in placements:
        bounds = tuple(round(v*scale) for v in (x,y,x+plate.w,y+plate.h))
        image.crop(bounds).save(OUT / (plate.name+".png"))
    image.resize((1200, round(H/3))).save(OUT / "overview-preview.png")
    print(f"PNG export passed: {image.width}x{image.height}; {len(plates)} plates; no overflowing node text")


if __name__ == "__main__":
    import sys
    if "--png" in sys.argv:
        render_png()
