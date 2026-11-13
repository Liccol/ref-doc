"""MSA Index Branch: projections, cache, block scores and selected indices."""
import json
from pathlib import Path
from diagram_primitives import Plate,STYLE
ROOT=Path(__file__).resolve().parent
W,H=2800,1610
p=Plate('msa-index-branch',2680,1280,'MSA Index Branch：为每个 query、每个索引组选择历史块','orange')
# Bottom-up computation, with cache and teaching example alongside.
p.box(130,1110,1370,['输入 X [N,6144] + 位置 positions [N]'],68)
p.box(130,915,1370,['Index Q / K 线性投影 → GemmaRMSNorm → partial RoPE',
    'Q_idx [N,4,128]；K_idx [N,1,128]，共享 1 路 index K'],102)
p.arrow([(815,1110),(815,1017)])
p.box(1700,915,850,['Index K Cache [T,1,128]（逻辑视图）',
    '写入当前 K_idx，保留已有历史 K_idx',
    'T 为当前可见历史长度；按分页缓存存储'],137)
p.arrow([(1500,966),(1700,966)])
p.box(130,680,1370,['逐组索引评分：s_g(t,j) = Q_idx[t,g] · K_idx[j]',
    '每个 query 有 4 路分数；逻辑 shape [N,4,T]',
    '屏蔽未来位置；实现可融合计算，无需物化完整分数矩阵'],137)
p.arrow([(815,915),(815,817)])
p.arrow([(2125,915),(2125,850),(1550,850),(1550,748),(1500,748)])
p.box(130,475,1370,['Block Max Pooling：把 token 分数变成块分数',
    'S_g(t,b) = max（块 b 内有效位置的 s_g(t,j)）',
    '每 128 个位置一块；逻辑 shape [N,4,ceil(T/128)]'],137)
p.arrow([(815,680),(815,612)])
p.box(130,290,1370,['各组独立选 Top-16 blocks + 强制 local block',
    'local=1，占用 Top-16 预算；短序列受有效块数限制'],102)
p.arrow([(815,475),(815,392)])
p.box(130,105,1370,['输出 block 索引 [N,4,16]（逻辑容量）',
    '供主分支读取对应 K/V 块；索引分支不生成最终 attention 输出'],102)
p.arrow([(815,290),(815,207)])
p.box(1700,110,850,['与普通因果 GQA 的关系',
    '普通 GQA：读取全部可见历史 K/V',
    'MSA：增加 Index Branch，先选块',
    '主分支再在选中块上计算注意力'],172)
p.text(2125,360,'选块教学例：只画 1 个 query 的 1 路索引','node','middle')
p.text(2125,405,'假设每块 2 个位置，预算为 2 块','body','middle')
for b,score in enumerate(['0.3','0.9','0.2','0.4','0.5','local']):
 x=1725+b*133
 p.rect(x,445,120,85,'#dfb56e' if b in (1,5) else '#edf0f4')
 p.text(x+60,480,f'B{b}','node','middle')
 p.text(x+60,512,score,'small','middle')
p.text(2125,585,'保留 B1 + 当前 local B5；每块整体读取','body','middle')
p.text(2125,635,'其他索引组独立评分，可选出不同块','body','middle')
p.text(2125,715,'真实配置：block=128，4 路，Top-16 / 路','body','middle')
p.text(2125,765,'分数与选块示例为教学数值，不是运行结果','small','middle')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
 f'<rect width="{W}" height="{H}" fill="white"/>',STYLE,
 '<text x="1400" y="65" class="title" text-anchor="middle">MiniMax-M3.1 · MSA Index Branch</text>',
 '<text x="1400" y="115" class="body" text-anchor="middle">自下向上：索引投影 → token 评分 → 块内 max → 按组 Top-K → block 索引</text>',
 '<g transform="translate(60,170)">'+''.join(p.items)+'</g>',
 '<text x="1400" y="1510" class="body" text-anchor="middle">Index Q/K 与主 attention Q/K 使用独立参数；Index K Cache 也与主 K/V Cache 分开。</text>',
 '<text x="1400" y="1560" class="small" text-anchor="middle">N 为当前批次参与计算的 token 数；所有 shape 为未分片逻辑维度，缓存的实际布局与融合方式依后端实现。</text></svg>']
(ROOT/'msa-index-branch.svg').write_text('\n'.join(svg),encoding='utf-8')

def render_png():
    import html,os,re,subprocess,tempfile
    from PIL import Image
    candidates=[Path(os.environ.get('PROGRAMFILES','C:/Program Files'))/'Google/Chrome/Application/chrome.exe',
                Path(os.environ.get('PROGRAMFILES(X86)','C:/Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe']
    browser=next(p for p in candidates if p.exists())
    qa="""<script>const bad=[];document.querySelectorAll('text[data-max-width]').forEach(t=>{if(t.getBBox().width>+t.dataset.maxWidth)bad.push(t.textContent)});const p=document.createElement('pre');p.id='qa';p.textContent=JSON.stringify(bad);document.body.append(p)</script>"""
    with tempfile.TemporaryDirectory(prefix='msa-index-') as td:
        temp=Path(td);page=temp/'index.html'
        page.write_text('<!doctype html><meta charset="utf-8"><style>body{margin:0}</style>'+''.join(svg)+qa,encoding='utf-8')
        r=subprocess.run([str(browser),'--headless','--disable-gpu','--hide-scrollbars','--no-first-run',
            '--user-data-dir='+str(temp/'profile'),'--force-device-scale-factor=1.5',f'--window-size={W},{H}',
            '--screenshot='+str(ROOT/'msa-index-branch.png'),'--dump-dom',page.as_uri()],capture_output=True,text=True,encoding='utf-8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,r.stderr[-1000:]
        report=re.search(r'<pre id="qa">(.*?)</pre>',r.stdout,re.S)
        assert report and not json.loads(html.unescape(report.group(1))),report.group(1) if report else 'No QA report'
    im=Image.open(ROOT/'msa-index-branch.png');assert im.size==(4200,2415)
    im.resize((1400,805)).save(ROOT/'msa-index-branch-preview.png')
    print('MSA Index Branch rendered at 4200x2415; node text fits.')

if __name__=='__main__':
    import sys
    if '--png' in sys.argv:render_png()
