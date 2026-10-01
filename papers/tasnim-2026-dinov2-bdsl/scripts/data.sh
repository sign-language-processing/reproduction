#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import collections, hashlib, json, pathlib, shutil, urllib.request, zipfile
root=pathlib.Path('/datasets/bdsl49-v6-recognition');root.mkdir(parents=True,exist_ok=True)
if (root/'manifest.json').exists():
 print((root/'manifest.json').read_text());raise SystemExit()
urls={'Recognition_1.zip':'https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/d5f2b6cd-5567-4c19-94c9-f51ef7add9b1','Recognition_2.zip':'https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/5c1ce9bc-2bfc-4878-8565-04d07be32a8e'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
expected={'Recognition_1.zip':'02fd87ed474798b617865c4fdfa49823dd3aa9f186f5eb96681c2344b42b3ff6','Recognition_2.zip':'28be109329fa794b0a9ad1dc027f8f17868a6573344db1a330e280db3ba61c4a'}
archives={}
for name,url in urls.items():
 path=root/name
 if not path.exists():
  print('Download',name,flush=True)
  with urllib.request.urlopen(url,timeout=120) as r,(root/(name+'.partial')).open('wb') as f:shutil.copyfileobj(r,f)
  (root/(name+'.partial')).rename(path)
 digest=sha(path)
 assert digest==expected[name],name
 archives[name]={'url':url,'bytes':path.stat().st_size,'sha256':digest}
 with zipfile.ZipFile(path) as z:
  for entry in z.infolist():
   target=(root/entry.filename).resolve()
   if not target.is_relative_to(root.resolve()):raise RuntimeError('Unsafe archive path')
  z.extractall(root)
counts={};files=[]
for p in sorted(root.rglob('*')):
 if p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
 rel=p.relative_to(root);parts=rel.parts
 split=next((s for s in ('train','test') if s in parts),None)
 if not split:raise RuntimeError(str(rel))
 label=parts[parts.index(split)+1]
 counts.setdefault(split,collections.Counter())[label]+=1
 files.append((str(rel),sha(p)))
print('Counts', {s:dict(c) for s,c in counts.items()},flush=True)
assert sum(counts['train'].values())==11774 and sum(counts['test'].values())==2940
assert len(counts['train'])==len(counts['test'])==49
(root/'files.sha256').write_text(''.join(h+'  '+p+'\n' for p,h in files))
manifest={'dataset':'BDSL49','version':6,'subset':'Recognition_1 and Recognition_2 original train/test','source':'https://data.mendeley.com/datasets/k5yk4j8z8s/6','doi':'10.17632/k5yk4j8z8s.6','license':'CC BY 4.0','archives':archives,'class_counts':{s:dict(c) for s,c in counts.items()},'counts':{s:sum(c.values()) for s,c in counts.items()},'files_sha256':sha(root/'files.sha256')}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
PY
