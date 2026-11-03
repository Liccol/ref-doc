"""Rebuild editable architecture SVG; all dimensions are logical, before sharding."""
from pathlib import Path
from html import escape
P=Path(__file__).resolve().parent
S=['<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2780" viewBox="0 0 2400 2780">', '<rect width="2400" height="2780" fill="white"/>', '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#405066"/></marker></defs>', '<style>text{font-family:Microsoft YaHei,Arial,sans-serif;fill:#21334a}.title{font-size:44px;font-weight:700}.head{font-size:26px;font-weight:700}.body{font-size:21px}.small{font-size:18px}.tiny{font-size:16px}</style>']
def text(x,y,t,cls='body',anchor='start'):
    S.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(t)}</text>')
def panel(x,y,w,h,title,color):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{color}" fill-opacity="0.045" stroke="{color}" stroke-width="2" stroke-dasharray="9 6"/>');text(x+20,y+38,title,'head')
def box(x,y,w,h,lines,color='#e8eef6'):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{color}" stroke="#9cacc0"/>')
    if isinstance(lines,str):lines=[lines]
    for i,l in enumerate(lines):text(x+w/2,y+h/2-(len(lines)-1)*13+7+i*26,l,'body','middle')
def arrow(points,dash=False):
    S.append(f'<polyline points="{points}" fill="none" stroke="#405066" stroke-width="2" '+('stroke-dasharray="7 5" ' if dash else '')+'marker-end="url(#a)"/>')
def add(x,y):
    S.append(f'<circle cx="{x}" cy="{y}" r="18" fill="white" stroke="#405066" stroke-width="2"/>');text(x,y+7,'+','head','middle')
text(1200,68,'MiniMax-M3.1-Flash · 多模态 MoE 模型结构','title','middle')
text(1200,108,'60 层 MSA 主干 / hidden 6144 / GQA 64:4 / Top-4 of 128 experts / 32 层 ViT','body','middle')
# Center: actual layer order, top-down and explicit residuals.
panel(895,160,580,1390,'01 语言主干与单层展开','#7b61b0')
for y,ls in [(225,['文本 Token IDs → Embedding','词表 200064；输出维度 6144']),(355,['视觉 embedding 替换对应占位符','合并序列 [T, 6144]']),(510,['GemmaRMSNorm','ε = 1e-6']),(635,['MiniMax Sparse Attention','见左侧 MSA / indexer 展开']),(835,['GemmaRMSNorm','attention 残差相加后归一化']),(960,['FFN（按层选择）','层 0–2 Dense；层 3–59 MoE']),(1245,['最终 GemmaRMSNorm']),(1360,['LM Head：6144 → 200064','不与输入 embedding 绑权'])]:box(955,y,450,82,ls,'#f0eafa')
for y1,y2 in [(307,355),(437,510),(592,635),(917,960),(1327,1360)]:arrow(f'1180,{y1} 1180,{y2}')
add(1180,770);arrow('1180,717 1180,752');arrow('1180,788 1180,835');arrow('1180,475 925,475 925,770 1162,770')
add(1180,1100);arrow('1180,1042 1180,1082');arrow('1180,1100 1445,1100 1445,800 1180,800');arrow('1180,800 1420,800 1420,1100 1198,1100')
# Replace accidental ambiguous loop with explicit next-layer output / residual bypass.
S=S[:-2];arrow('1180,800 1420,800 1420,1100 1198,1100');arrow('1180,1118 1180,1170');text(1180,1202,'重复 60 层；输出送最终归一化','small','middle');arrow('1180,1209 1180,1245')
text(1428,605,'残差','tiny');text(1430,934,'残差','tiny')
arrow('1180,1442 1180,1490');text(1180,1520,'logits → sampling → 下一个 token','small','middle')
# Attention detail two independently connected branches.
panel(45,160,800,1390,'02 MSA：主 QKV 与稀疏索引双分支','#db8a23')
box(215,225,460,70,['归一化输入 X [T, 6144]'],'#fff0dc')
box(75,350,350,85,['主投影：6144 → 9216','Q:8192 / K:512 / V:512'],'#fff0dc')
box(465,350,350,85,['索引投影：6144 → 640','index Q:512 / index K:128'],'#fff0dc')
arrow('445,295 445,320 250,320 250,350');arrow('445,320 640,320 640,350')
box(75,495,350,100,['Q: [T,64,128]；K/V: [T,4,128]','Q/K per-head norm','Q/K RoPE：64/128 维'],'#fff0dc')
box(465,495,350,100,['index Q: [T,4,128]','index K: [T,1,128]','per-head norm + RoPE'],'#fff0dc')
arrow('250,435 250,495');arrow('640,435 640,495')
box(75,660,350,85,['分页主 KV cache','slot_mapping 写入 K 与 V'],'#ffe3dd')
box(465,660,350,85,['分页 index K cache','block_table 定位历史 block'],'#ffe3dd')
arrow('250,595 250,660');arrow('640,595 640,660')
box(465,815,350,110,['Index score → block Top-K','4 个 index heads，dim 128','score type = max'],'#fff0dc');arrow('640,745 640,815')
box(465,990,350,100,['Top-K 16 blocks，block 128','init blocks=0；local blocks=1','有效索引与因果边界处理'],'#fff0dc');arrow('640,925 640,990')
box(95,1165,700,100,['Sparse Attention：Q × selected Kᵀ → softmax → V','GQA：每个 KV head 对应 16 个 Q heads','输出 [T,64,128] → [T,8192]'],'#ffdbab')
arrow('640,1090 640,1165');arrow('250,745 250,1165');arrow('75,545 60,545 60,1215 95,1215')
box(215,1340,460,75,['O projection：8192 → 6144'],'#fff0dc');arrow('445,1265 445,1340');arrow('675,1378 865,1378 865,677 955,677',True)
text(80,1470,'逻辑投影；物理实现可融合 QKV 与 index 投影。','small')
text(80,1503,'全 60 层使用稀疏 attention；没有 MLA latent KV。','small')
# MoE + activation detailed branch.
panel(1525,160,830,820,'03 MoE：后 57 层的 FFN','#3c89bd')
box(1715,225,450,70,['输入 [T,6144]'],'#e3f2fc')
box(1560,365,365,85,['Router linear：6144 → 128','sigmoid + correction bias'],'#e3f2fc')
box(1970,365,350,85,['1 个 shared expert','每个 token 都参与'],'#e4f2df')
arrow('1940,295 1940,320 1740,320 1740,365');arrow('1940,320 2145,320 2145,365')
box(1560,520,365,85,['128 routed experts 中选 Top-4','归一化路由权重'],'#e3f2fc');arrow('1740,450 1740,520')
box(1560,675,365,85,['4 个选中 expert 输出加权求和','routed scaling factor = 2.0'],'#e3f2fc');arrow('1740,605 1740,675')
box(1970,520,350,85,['Expert FFN','6144 → 3072 → 6144'],'#e4f2df');arrow('2145,450 2145,520')
add(1940,855);arrow('1740,760 1740,855 1922,855');arrow('2145,605 2145,855 1958,855');arrow('1940,873 1940,920');text(1940,950,'输出 [T,6144] → 层残差相加','small','middle')
arrow('1405,1000 1500,1000 1500,260 1715,260',True)
panel(1525,1020,830,530,'04 Dense / Expert FFN 内部','#63a249')
box(1755,1090,365,65,['X [T,6144]'],'#e4f2df')
box(1560,1210,350,80,['Linear gate：6144 → I','SwiGLUOAI 门控分支'],'#e4f2df');box(1970,1210,350,80,['Linear up：6144 → I','value 分支'],'#e4f2df')
arrow('1940,1155 1940,1180 1735,1180 1735,1210');arrow('1940,1180 2145,1180 2145,1210')
box(1755,1350,365,70,['逐元素门控组合'],'#e4f2df');arrow('1735,1290 1735,1385 1755,1385');arrow('2145,1290 2145,1385 2120,1385')
box(1755,1460,365,60,['Linear down：I → 6144'],'#e4f2df');arrow('1940,1420 1940,1460');text(1555,1540,'I：Dense 12288；routed/shared expert 3072','small')
# Lower vision route with true projector-before-merge order.
panel(45,1610,2310,930,'05 视觉与语言融合：ViT → Projector → Patch Merger → token 替换','#388f86')
for x,w,ls in [(80,340,['图像 / 视频预处理','dynamic resolution','pixel_values + grid_thw']),(485,375,['Conv3d patch embedding','kernel / stride = (2,14,14)','输出 [Nv,1280]']),(925,420,['Pre LayerNorm','Vision Transformer × 32','hidden 1280；heads 16；dim 80']),(1410,420,['多模态 projector','1280 → 6144 → 6144','GELU；bias=True']),(1895,420,['2×2 spatial patch merge','24576 → 6144 → 6144','GELU；输出 [Nv/4,6144]'])]:box(x,1705,w,125,ls,'#e2f3ef')
for x1,x2 in [(420,485),(860,925),(1345,1410),(1830,1895)]:arrow(f'{x1},1767 {x2},1767')
arrow('2105,1830 2330,1830 2330,1575 870,1575 870,396 955,396',True)
panel(80,1900,1080,550,'视觉 Transformer 单层 × 32','#388f86')
for y,ls in [(1970,['LayerNorm → QKV linear：1280 → 3840']),(2080,['Q/K 3D RoPE → MHA（16 heads × 80）→ O linear']),(2220,['LayerNorm → MLP：1280 → 5120 → 1280','GELU'])]:box(170,y,880,75,ls,'#e2f3ef')
arrow('610,2045 610,2080');add(610,2180);arrow('610,2155 610,2162');arrow('610,2198 610,2220');arrow('170,2005 120,2005 120,2180 592,2180')
add(610,2360);arrow('610,2295 610,2342');arrow('610,2200 1100,2200 1100,2360 628,2360');text(610,2420,'Pre-norm + 两次残差；不加载未训练 post LayerNorm','small','middle')
panel(1215,1900,1100,550,'参数、布局与证据边界','#8392a5')
for i,l in enumerate(['视觉：3D RoPE θ=10000；temporal patch=2；spatial merge=2。','文本：RoPE θ=10,000,000，rotary dim=64；GemmaRMSNorm ε=1e-6。','文本输出：attention_output_gate=False；不是 gated MLA。','语言层 0–2：MSA + Dense FFN；层 3–59：MSA + MoE。','Nv 为视觉 patch token 数；T 为合并后的语言 token 数。','图示为逻辑算子流，实际内核可能融合；非设备时间线。','官方公开配置包含 MXFP8 字段；本图不改变实测 BF16 验收口径。','本次只核对配置与实现、绘图，没有启动 NPU 或重跑精度。']):text(1240,1985+i*53,l,'small')
text(80,2610,'来源：官方 MiniMaxAI/MiniMax-M3.1-Flash config（固定 revision 54e4e69）及本地 MiniMax-M3 适配代码。','body')
text(80,2650,'布局参考 CalvinXKY/InfraTech · models/kimi_k_3；模块、参数与连线按 MiniMax 重新绘制。','body')
text(80,2690,'箭头：实线为数据流 / 残差；虚线为模块展开或跨区引用。维度为逻辑维度，不是 TP/EP 分片后的尺寸。','body')
S.append('</svg>');(P/'minimax-m31-full-architecture.svg').write_text('\n'.join(S),encoding='utf8')
