"""Standalone comparison of the DSA and MSA index branches only."""

import json
from pathlib import Path
from diagram_primitives import Plate,STYLE

ROOT=Path(__file__).resolve().parent
W,H=2800,1610

def panel(msa):
    name='MSA / MiniMax-M3.1' if msa else 'DSA / DeepSeek-V3.2'
    p=Plate(name,1300,1280,name,'orange' if msa else 'blue')
    p.text(60,100,'Index Branch','node')
    p.rect(70,125,1160,280,'#fbfcfe','#c6d1de',10)
    p.text(650,170,'输出：Top-K block 索引' if msa else '输出：Top-K token 索引','node','middle')
    p.text(650,210,'只画其中一路索引；M3.1 实际有 4 路独立选块' if msa else '各索引头先聚合，最终产生一组 token 索引','small','middle')
    selected={2,3,10,11} if msa else {1,4,7,10}
    if msa:
        for block in range(6):
            x=184+block*164
            p.rect(x,238,152,86,'#ffffff','#98a4b5',8)
            p.text(x+76,350,f'B{block}','small','middle')
    for pos in range(12):
        x=190+pos*82
        p.rect(x,250,64,56,('#dfb56e' if msa else '#93b4d0') if pos in selected else '#edf0f4')
        p.text(x+32,286,f'{pos:02}','small','middle')
    p.text(650,385,'例：选 B1 + local B5，包含块内全部 token' if msa else '例：选位置 01、04、07、10','small','middle')
    p.box(230,465,840,[
        'Block Top-K + local 必选' if msa else 'Token Top-K',
        '按块分数排序，保留重要块和当前 local 块' if msa else '按 token 分数排序，保留重要的单个位置'],102)
    p.arrow([(650,465),(650,405)])
    p.box(230,655,840,[
        '逐路点积 → Block Max Pooling' if msa else '多头点积 → ReLU → 加权聚合',
        'score_g(block) = max_j∈block (q_g · k_j)' if msa else 'score(token j) = Σ_h w_h · ReLU(q_h · k_j)',
        '每个索引组分别得到每个 block 的分数' if msa else '多个索引头合成每个 token 的一个分数'],137)
    p.arrow([(650,655),(650,567)])
    p.box(230,880,840,[
        '独立 index Q/K 投影 → Norm & RoPE' if msa else 'Index Q/K 投影 + query 相关的 head weights',
        '4 路 index Q，共享 1 路 index K（M3.1）' if msa else '多索引头参与评分，由权重决定如何聚合'],102)
    p.arrow([(650,880),(650,792)])
    p.box(230,1100,840,['输入 hidden state + 历史 index key'],68)
    p.arrow([(650,1100),(650,982)])
    return p

left,right=panel(False),panel(True)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',STYLE,
     '<text x="1400" y="65" class="title" text-anchor="middle">DSA 与 MSA：只看 Index Branch 的评分与选择</text>',
     '<text x="1400" y="115" class="body" text-anchor="middle">自下向上：输入 → 索引投影 → 评分 → Top-K → 索引输出；不展开后续主注意力</text>',
     '<g transform="translate(60,170)">'+''.join(left.items)+'</g>',
     '<g transform="translate(1440,170)">'+''.join(right.items)+'</g>',
     '<text x="1400" y="1510" class="body" text-anchor="middle">区别：DSA 把多索引头聚合成 token 分数；MSA 逐路计算块分数，输出 block 索引。</text>',
     '<text x="1400" y="1560" class="small" text-anchor="middle">选取示例采用 12 个历史位置、2-token 块和 4 个候选位置，仅用于示意；M3.1 实际 block=128、Top-16 / 路，local 占预算。</text></svg>']
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
    im=Image.open(ROOT/'msa-vs-dsa.png');assert im.size==(4200,2415)
    im.resize((1400,805)).save(ROOT/'msa-vs-dsa-preview.png')
    print('Standalone comparison rendered at 4200x2415; node text fits.')

if __name__=='__main__':
    import sys
    if '--png' in sys.argv:render_png()
