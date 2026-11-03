import json, os, re, sys, glob
json_dir, img_dir = sys.argv[1], sys.argv[2]
names = set()
# Only the val dataset (textvqa_val.jsonl) is used by the textvqa_gen
# evaluation scenario. Process only it so the 'needed' image count
# reflects the actual eval subset, not the much larger train set.
for fp in glob.glob(os.path.join(json_dir, '*val*.jsonl')):
    lines = open(fp, encoding='utf-8').read().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            out.append(ln)
            continue
        if isinstance(d, dict) and 'image' in d:
            names.add(os.path.basename(d['image']))
            d['image'] = os.path.join(img_dir, os.path.basename(d['image'])).replace('\\', '/')
            out.append(json.dumps(d, ensure_ascii=False))
        else:
            out.append(ln)
    open(fp, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
pat = re.compile(r'data/textvqa/train_images/([A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp))')
for fp in glob.glob(os.path.join(json_dir, '*.json')):
    if fp.endswith('.bak'):
        continue
    txt = open(fp, encoding='utf-8').read()
    txt2 = pat.sub(lambda m: os.path.join(img_dir, m.group(1)).replace('\\', '/'), txt)
    if txt2 != txt:
        open(fp, 'w', encoding='utf-8').write(txt2)
with open(os.path.join(json_dir, '.image_list.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(sorted(names)))
print('textvqa image paths absolutized; referenced images:', len(names))