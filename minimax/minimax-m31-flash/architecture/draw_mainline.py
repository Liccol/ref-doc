"""One primary model path with connected module insets. Run with --png."""

import json
from pathlib import Path
from diagram_primitives import Plate, STYLE, COLORS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mainline'
OUT.mkdir(exist_ok=True)
cfg=json.loads((ROOT/'sources/config.json').read_text(encoding='utf-8'))['text_config']
assert cfg['num_hidden_layers']==60 and sum(cfg['moe_layer_freq'])==57
W,H=3400,3260

# The one dominant path. Only module names belong inside its decoder blocks.
p=Plate('main-path',1080,3020,'主线：输入 → 多模态合并 → 60 层 → 输出','purple')
p.box(180,100,720,['输入：文本 / 图像 / 视频'],68)
p.arrow([(540,168),(540,205),(315,205),(315,240)])
p.arrow([(540,205),(840,205),(840,240)])
p.box(90,240,450,['Tokenizer','文本分词 + 视觉占位展开'],102,COLORS['blue'][1])
p.box(650,240,380,['视觉预处理','pixels + grid_thw'],102,COLORS['teal'][1])
p.down(315,342,380);p.down(840,342,380)
p.box(90,380,450,['Embedding','token ID → 6144 维向量'],102,COLORS['blue'][1])
p.box(650,380,380,['视觉塔 ①','→ 6144 维视觉向量'],102,COLORS['teal'][1])
p.arrow([(315,482),(315,530),(540,530),(540,560)])
p.arrow([(840,482),(840,530),(540,530)])
p.box(180,560,720,['按占位位置合并为一条序列','语言向量 + 视觉向量 → X [N,6144]'],102)
p.down(540,662,720)

def decoder(top,title,ffn,label):
    # Neutral dashed outline is a repeat marker, rather than a second flow.
    p.items.append(f'<rect x="180" y="{top}" width="720" height="730" rx="16" fill="none" stroke="#98a4b5" stroke-width="2" stroke-dasharray="10 7"/>')
    p.text(540,top+43,title,'node','middle')
    for dy,txt,color in [(90,'GemmaRMSNorm ②','purple'),(215,'Attention / MSA ③','orange'),(385,'GemmaRMSNorm ②','purple'),(510,ffn+' '+label,'green')]:
        p.box(280,top+dy,520,[txt],68,COLORS[color][1])
    for y1,y2 in [(top+70,top+90),(top+158,top+215),(top+283,top+325),(top+365,top+385),(top+453,top+510),(top+578,top+630)]:p.down(540,y1,y2)
    p.add(540,top+345);p.add(540,top+650)
    p.arrow([(540,top+70),(225,top+70),(225,top+345),(520,top+345)])
    p.arrow([(540,top+365),(855,top+365),(855,top+650),(560,top+650)])
    p.down(540,top+670,top+730)

decoder(720,'前三层：层 0–2，重复 ×3','Dense FFN','④')
p.down(540,1450,1510)
decoder(1510,'后 57 层：层 3–59，重复 ×57','MoE','⑤')
p.down(540,2240,2310)
p.box(180,2310,720,['Final GemmaRMSNorm ②'],68)
p.down(540,2378,2440)
p.box(180,2440,720,['LM Head','6144 → 200064 维 logits'],102)
p.down(540,2542,2600)
p.box(180,2600,720,['Sampling → 新 token ID'],68)
p.down(540,2668,2730)
p.box(180,2730,720,['文本输出'],68)
p.text(540,2865,'层内只列关键模块，细节沿虚线看右侧。','small','middle')
p.text(540,2905,'N 为合并后的有效 token 数；两类位置走同一主干。','small','middle')
spine=p

# Inset 1: the vision branch only; its final vector returns to the merge node.
p=Plate('01-vision',2060,540,'① 视觉塔：把图片 / 视频变成语言模型能接收的向量','teal')
for x,ww,lines in [
    (35,450,['Patch Embedding','Conv3d：2×14×14 patch','[Nv,1176] → [Nv,1280]']),
    (550,450,['ViT ×32','LN → MHA → 残差','LN → GELU MLP → 残差']),
    (1065,450,['Projector','Linear → GELU → Linear','1280 → 6144 → 6144']),
    (1580,445,['2×2 Patch Merger','4 个相邻向量拼接后映射','[Nv/4,24576] → [Nv/4,6144]'])]:p.box(x,120,ww,lines,137)
for a,b in [(485,550),(1000,1065),(1515,1580)]:p.arrow([(a,188),(b,188)])
p.box(35,315,1990,['输出 [Nv/4,6144]，按图像/视频占位位置替换输入 Embedding 中的对应向量。','例：处理后 672×672 图片 → 2304 个 patch → 576 个视觉 token；实际数量由 grid 决定。','视觉塔独立于 60 层语言主干；合并后不再按语言/视觉分成两条主干。'],137,'#f3f6fa')
vision=p

p=Plate('02-gemma-rmsnorm',2060,230,'② GemmaRMSNorm：归一化幅度，再做可学习的逐通道缩放','purple')
p.text(50,125,'y = x / sqrt(mean(x²) + ε) × (1+w)；ε=10⁻⁶；不减均值，输出 shape 不变。','body')
p.text(50,177,'层内沿最后 6144 维计算；主 Q/K 和 index Q/K 则每 head 沿 128 维计算，权重各自独立。','body')
norm=p

p=Plate('03-attention-msa',1080,1450,'③ Attention：MSA 双分支选块与计算','orange')
p.box(230,110,620,['输入 H [N,6144]'],68)
p.arrow([(540,178),(540,215),(280,215),(280,250)])
p.arrow([(540,215),(800,215),(800,250)])
p.box(40,250,480,['主 Q / K / V Linear','Q [N,64,128]','K/V 各 [N,4,128]'],137)
p.box(580,250,460,['index Q / K Linear','iQ [N,4,128]','iK [N,1,128]'],137)
p.down(280,387,430);p.down(800,387,430)
p.box(40,430,480,['Q/K Norm + Partial RoPE','每头前 64 维旋转','V 保持原值'],137)
p.box(580,430,460,['iQ/iK Norm + Partial RoPE','位置编码后参与索引打分','iK 按请求写入缓存'],137)
p.down(800,567,610)
p.box(580,610,460,['Index Score → Block Top-K','块内点积取 max，再选 Top-16','block=128；强制 local=1'],137)
p.box(40,650,480,['主 K/V 缓存','新 K/V 写入；Q 不缓存','旧历史保留，按所选块读取'],137,COLORS['red'][1])
p.arrow([(280,567),(280,650)])
p.box(100,860,880,['选中历史块 → 主 GQA Attention','QKᵀ / √128 → causal mask → softmax → V','64 Q heads : 4 KV heads；每组 16 个 Q heads'],137)
p.arrow([(800,747),(800,860)]);p.arrow([(280,787),(280,860)])
# The current Q bypasses the cache, making its data path explicit.
p.arrow([(40,498),(20,498),(20,930),(100,930)])
p.down(540,997,1040)
p.box(100,1040,880,['合并 heads → O Linear','[N,8192] → [N,6144]'],102)
p.box(40,1200,1000,['MSA 的关键：先找块，再在选中范围内精算注意力。','4 个 KV 组可选不同块；local 块占 Top-16 预算。','稀疏的是主注意力读取范围，不是直接删除未选缓存。'],137,'#f3f6fa')
msa=p

p=Plate('04-dense-ffn',900,650,'④ Dense FFN：每 token 用同一套本层参数','green')
p.box(260,105,380,['Z [N,6144]'],68)
p.arrow([(450,173),(450,205),(235,205),(235,240)])
p.arrow([(450,205),(665,205),(665,240)])
p.box(35,240,400,['Gate Linear','[N,12288]'],102)
p.box(465,240,400,['Up Linear','[N,12288]'],102)
p.box(155,400,590,['SwiGLU-OAI 门控相乘','g′·sigmoid(1.702g′)·(u′+1)'],102)
p.arrow([(235,342),(235,400)]);p.arrow([(665,342),(665,400)])
p.down(450,502,540)
p.box(155,540,590,['Down Linear → [N,6144]'],68)
dense=p

p=Plate('05-moe',900,1280,'⑤ MoE：路由专家 + 无条件共享专家','green')
p.box(230,105,440,['同一份 Z [N,6144]'],68)
p.arrow([(450,173),(450,205),(235,205),(235,240)])
p.arrow([(450,205),(680,205),(680,240)])
p.box(35,240,400,['Router Linear → sigmoid','128 维得分，加 bias 选 Top-4','IDs / weights 各 [N,4]'],137)
p.box(485,240,380,['Shared Expert','每个 token 都参与','不占 Top-4 名额'],137,COLORS['teal'][1])
p.down(235,377,420)
p.box(35,420,400,['Dispatch → 选中的 4 个专家','按 expert ID 分组计算','共 4N 条 token-expert 任务'],137)
p.down(235,557,600)
p.box(35,600,400,['Expert FFN','gate/up → SwiGLUOAI → down','6144 → 3072 → 6144'],137)
p.arrow([(680,377),(680,600)])
p.box(485,600,380,['Shared FFN','gate/up → 激活 → down','6144 → 3072 → 6144'],137,COLORS['teal'][1])
p.down(235,737,780)
p.box(35,780,400,['Combine：回收 + 加权求和','R=2.0 × Σ rₖ·Expertₖ(Z)','r 取原始 sigmoid 分数归一化'],137)
p.add(450,970)
p.arrow([(235,917),(235,970),(430,970)])
p.arrow([(680,737),(680,970),(470,970)])
p.down(450,990,1030)
p.box(130,1030,640,['F=R+Shared(Z) → [N,6144]'],68)
p.text(450,1160,'语言和视觉 token 均走两条分支。','small','middle')
p.text(450,1205,'Dense / MoE 的区别在 FFN；二者的 Attention 都是 MSA。','small','middle')
moe=p

def bottom_up(plate, low, high):
    """Reverse diagram geometry while keeping text upright and box line order."""
    import xml.etree.ElementTree as ET
    axis=low+high
    nodes=[ET.fromstring(item) for item in plate.items]
    original=[ET.fromstring(item) for item in plate.items]
    boxes=[n for n in original[3:] if n.tag=='rect' and low<=float(n.get('y')) and float(n.get('y'))+float(n.get('height'))<=high]
    circles=[n for n in original[3:] if n.tag=='circle']
    result=plate.items[:3]
    for n in nodes[3:]:
        if n.tag=='rect':
            y,h=float(n.get('y')),float(n.get('height'))
            if low<=y and y+h<=high:n.set('y',str(axis-y-h))
        elif n.tag=='circle':
            y=float(n.get('cy'))
            if low<=y<=high:n.set('cy',str(axis-y))
        elif n.tag=='polyline':
            points=[tuple(map(float,s.split(','))) for s in n.get('points').split()]
            if all(low<=y<=high for _,y in points):n.set('points',' '.join(f'{x},{axis-y}' for x,y in points))
        elif n.tag=='text':
            x,y=float(n.get('x')),float(n.get('y'))
            if low<=y<=high:
                if n.text=='+':
                    circle=min(circles,key=lambda c:abs(float(c.get('cx'))-x)+abs(float(c.get('cy'))+8-y))
                    n.set('y',str(axis-float(circle.get('cy'))+8))
                else:
                    enclosing=[b for b in boxes if float(b.get('x'))<=x<=float(b.get('x'))+float(b.get('width')) and float(b.get('y'))<=y<=float(b.get('y'))+float(b.get('height'))]
                    if enclosing:
                        box=min(enclosing,key=lambda b:float(b.get('width'))*float(b.get('height')))
                        old=float(box.get('y'));h=float(box.get('height'))
                        n.set('y',str(axis-old-h+(y-old)))
                        if n.text and '重复 ×' in n.text:
                            labels=['前三层','层 0–2','×3'] if n.text.startswith('前三层') else ['后 57 层','层 3–59','×57']
                            n.set('x','165');n.set('text-anchor','end');n.set('class','small');n.text=None
                            for i,label in enumerate(labels):
                                span=ET.SubElement(n,'tspan',{'x':'165','dy':'0' if i==0 else '34'})
                                span.text=label
                    else:n.set('y',str(axis-y))
        result.append(ET.tostring(n,encoding='unicode'))
    plate.items=result

bottom_up(spine,100,2798)
bottom_up(dense,105,608)
bottom_up(msa,110,1142)
bottom_up(moe,105,1098)

# One path on the left, explanatory insets on the right. Dashed = magnification.
placements=[(spine,60,150),(vision,1220,2520),(norm,1220,2220),
            (dense,1220,1500),(msa,2200,720),(moe,1220,170)]
full=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
      '<title>MiniMax-M3.1-Flash：一条主线与关键模块展开</title>',
      f'<rect width="{W}" height="{H}" fill="white"/>',STYLE,
      '<text x="1700" y="65" class="title" text-anchor="middle">MiniMax-M3.1-Flash · 主线与关键模块展开</text>',
      '<text x="1700" y="115" class="body" text-anchor="middle">自下向上：输入 → 合并 → 前 3 层 → 后 57 层 → 输出　｜　虚线为模块放大说明</text>']
for plate,x,y in placements:
    full.append(f'<g transform="translate({x},{y})">'+"\n".join(plate.items)+'</g>')

def zoom(points):
    full.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#8392a5" stroke-width="2" stroke-dasharray="9 7"/>')

zoom([(1090,2617),(1220,2617)])               # Vision tower → vision inset
zoom([(860,2205),(1140,2205),(1140,2335),(1220,2335)])
zoom([(860,1289),(1090,1289),(1090,1475),(2200,1475)])
zoom([(860,1784),(1220,1784)])                # Dense FFN → dense inset
zoom([(860,994),(1220,994)])                  # MoE → routed/shared inset
full.append('<text x="60" y="3205" class="note">逻辑 shape，未切 TP/EP；W=[out,in]。GemmaRMSNorm / MSA / SwiGLU-OAI 均按 M3.1 配置与本地适配实现解释，非实际运行 trace。</text>')
full.append('</svg>')
(ROOT/'minimax-m31-detailed-architecture.svg').write_text('\n'.join(full),encoding='utf-8')
for plate,_,_ in placements:(OUT/(plate.name+'.svg')).write_text(plate.render(),encoding='utf-8')
(OUT/'layout.json').write_text(json.dumps({'width':W,'height':H,'plates':[{'name':p.name,'x':x,'y':y,'width':p.w,'height':p.h} for p,x,y in placements]},indent=2)+'\n',encoding='utf-8')

def render_png():
    import html,os,re,subprocess,tempfile
    from PIL import Image
    browsers=[Path(os.environ.get('PROGRAMFILES','C:/Program Files'))/'Google/Chrome/Application/chrome.exe',
              Path(os.environ.get('PROGRAMFILES(X86)','C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe']
    browser=next(b for b in browsers if b.exists())
    script="""<script>const bad=[];document.querySelectorAll('text[data-max-width]').forEach(t=>{if(t.getBBox().width>+t.dataset.maxWidth)bad.push({text:t.textContent,width:t.getBBox().width,max:+t.dataset.maxWidth})});const p=document.createElement('pre');p.id='qa';p.textContent=JSON.stringify(bad);document.body.append(p)</script>"""
    with tempfile.TemporaryDirectory(prefix='minimax-mainline-') as td:
        work=Path(td);page=work/'render.html'
        page.write_text('<!doctype html><meta charset="utf-8"><style>body{margin:0}</style>'+''.join(full)+script,encoding='utf-8')
        result=subprocess.run([str(browser),'--headless','--disable-gpu','--hide-scrollbars','--no-first-run',
            '--user-data-dir='+str(work/'profile'),'--force-device-scale-factor=1.5',f'--window-size={W},{H}',
            '--screenshot='+str(ROOT/'minimax-m31-detailed-architecture.png'),'--dump-dom',page.as_uri()],capture_output=True,text=True,encoding='utf-8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,result.stderr[-1000:]
        report=re.search(r'<pre id="qa">(.*?)</pre>',result.stdout,re.S)
        assert report,'No browser layout report'
        assert not json.loads(html.unescape(report.group(1))),report.group(1)
    image=Image.open(ROOT/'minimax-m31-detailed-architecture.png')
    assert image.size==(5100,4890),image.size
    for plate,x,y in placements:
        image.crop(tuple(round(v*1.5) for v in (x,y,x+plate.w,y+plate.h))).save(OUT/(plate.name+'.png'))
    image.resize((1700,1630)).save(OUT/'preview.png')
    print('Mainline PNG 5100x4890; all inset text fits.')

if __name__=='__main__':
    import sys
    if '--png' in sys.argv:render_png()
