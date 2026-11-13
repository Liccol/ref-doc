"""Standalone DSA/MSA comparison with explicit Index and Main branches."""

import json
from pathlib import Path
from diagram_primitives import Plate,STYLE

ROOT=Path(__file__).resolve().parent
W,H=2800,2250

def panel(msa):
    name='MSA / MiniMax-M3.1' if msa else 'DSA / DeepSeek-V3.2'
    p=Plate(name,1300,1880,name,'orange' if msa else 'blue')
    p.box(70,110,1160,[
        'Main Branch：在选中的历史位置上计算 attention',
        'GQA：每组 Q heads 使用各自选中的块' if msa else 'MLA / MQA：同一 query 的所有 Q heads 使用同一组选点',
        '同组共享选块结果，不同组可以不同' if msa else '共享索引集合，不表示各 Q heads 的 attention 权重相同'],137)
    p.rect(35,285,1230,1080,'#ffffff','#b7c2cf',14)
    p.items[-1]=p.items[-1].replace('stroke-width="1.7"','stroke-width="1.7" stroke-dasharray="8 6"')
    p.text(60,325,'Index Branch','node')
    p.rect(70,355,1160,550,'#fbfcfe','#d0d8e2',10)
    p.text(650,405,'选择单位：整个 block' if msa else '选择单位：单个 token','node','middle')
    p.text(650,450,'示意采用相同预算：每个 Q head 最终读取 4 个历史位置','small','middle')
    for row,y in enumerate([530,685]):
        label=f'GQA组{row}' if msa else ('head 0' if row==0 else 'head 1…')
        p.text(85,y+36,label,'small')
        p.items[-1]=p.items[-1].replace('<text ','<text data-max-width="100" ')
        selected=({2,3,8,9} if row==0 else {0,1,8,9}) if msa else {1,4,7,10}
        color=('#dfb56e' if row==0 else '#a8bb8a') if msa else '#93b4d0'
        if msa:
            for block in range(6):
                x=184+block*164
                p.rect(x,y-12,152,86,'#ffffff','#98a4b5',8)
                p.text(x+76,y+95,f'B{block}','small','middle')
        for pos in range(12):
            x=190+pos*82
            p.rect(x,y,64,56,color if pos in selected else '#edf0f4')
            p.text(x+32,y+36,f'{pos:02}','small','middle')
    p.text(650,845,'每组各选 2 块 × 每块 2 token（教学示意）' if msa else '两行选中相同位置：一组 token 索引由各 Q heads 共用','small','middle')
    p.arrow([(650,355),(650,247)])
    p.box(230,1000,840,[
        '每组独立评分 → 块内 Max Pooling' if msa else '多索引头点积 → ReLU → 加权聚合',
        '先算 index Q · index K，再对块内位置取最大值' if msa else 'query 相关的权重把各索引头合成一个 token 分数',
        '得到“每组、每历史块”的分数' if msa else '得到“每个历史 token”的一个分数'],137)
    p.arrow([(650,1000),(650,905)])
    p.text(685,965,'Top-K blocks' if msa else 'Top-K tokens','small')
    p.box(230,1190,840,[
        '独立 index Q/K 投影 → Norm & RoPE' if msa else 'Index Q/K 投影 + query 相关的 head weights',
        'M3.1：4 个 index Q heads，共享 1 个 index K head' if msa else '多个索引头参与评分，随后聚合成一组选择结果'],102)
    p.arrow([(650,1190),(650,1137)])
    p.box(230,1480,840,['当前 query 的 hidden state + 历史上下文'],68)
    p.arrow([(650,1480),(650,1292)])
    p.box(70,1640,1160,[
        '结构特点：按块、按 GQA 组选择' if msa else '结构特点：细粒度 token 选择',
        'M3.1 实际配置：4 个组，每组 Top-16 个 128-token 块；local 必选。' if msa else '本图对比 V3.2 的 DSA：同一 query 共用一组 Top-K token 索引。',
        '每块整段读取；不同组能够关注不同的历史区域。' if msa else '允许从不同历史区域挑出分散的单个 token。'],137,'#f3f6fa')
    return p

left,right=panel(False),panel(True)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',STYLE,
     '<text x="1400" y="65" class="title" text-anchor="middle">DSA 与 MSA：Index Branch 到底选什么、由谁共享？</text>',
     '<text x="1400" y="115" class="body" text-anchor="middle">自下向上：输入 → 独立索引评分 → Top-K 选择 → Main Branch；数字是历史位置编号</text>',
     '<g transform="translate(60,170)">'+''.join(left.items)+'</g>',
     '<g transform="translate(1440,170)">'+''.join(right.items)+'</g>',
     '<text x="1400" y="2120" class="body" text-anchor="middle">共同点：两者都有 Index Branch；主注意力只在选中位置计算。区别：token / block，以及跨全部头共享 / 按 GQA 组共享。</text>',
     '<text x="1400" y="2180" class="small" text-anchor="middle">图中 12 个历史位置、2-token 块和 4 个位置的预算仅用于等预算示意；不是模型实测。MSA 实际 block=128。未对比速度。</text></svg>']
target=ROOT/'msa-vs-dsa.svg'
target.write_text('\n'.join(svg),encoding='utf-8')

def render_png():
    import html,os,re,subprocess,tempfile
    from PIL import Image
    candidates=[Path(os.environ.get('PROGRAMFILES','C:/Program Files'))/'Google/Chrome/Application/chrome.exe',
                Path(os.environ.get('PROGRAMFILES(X86)','C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe']
    browser=next(p for p in candidates if p.exists())
    qa="""<script>const bad=[];document.querySelectorAll('text[data-max-width]').forEach(t=>{if(t.getBBox().width>+t.dataset.maxWidth)bad.push(t.textContent)});const p=document.createElement('pre');p.id='qa';p.textContent=JSON.stringify(bad);document.body.append(p)</script>"""
    with tempfile.TemporaryDirectory(prefix='msa-dsa-') as td:
        temp=Path(td);page=temp/'index.html'
        page.write_text('<!doctype html><meta charset="utf-8"><style>body{margin:0}</style>'+''.join(svg)+qa,encoding='utf-8')
        r=subprocess.run([str(browser),'--headless','--disable-gpu','--hide-scrollbars','--no-first-run',
            '--user-data-dir='+str(temp/'profile'),'--force-device-scale-factor=1.5',f'--window-size={W},{H}',
            '--screenshot='+str(ROOT/'msa-vs-dsa.png'),'--dump-dom',page.as_uri()],capture_output=True,text=True,encoding='utf-8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,r.stderr[-1000:]
        report=re.search(r'<pre id="qa">(.*?)</pre>',r.stdout,re.S)
        assert report and not json.loads(html.unescape(report.group(1))),report.group(1) if report else 'No QA report'
    im=Image.open(ROOT/'msa-vs-dsa.png');assert im.size==(4200,3375)
    im.resize((1400,1125)).save(ROOT/'msa-vs-dsa-preview.png')
    print('Standalone comparison rendered at 4200x3375; node text fits.')

if __name__=='__main__':
    import sys
    if '--png' in sys.argv:render_png()
