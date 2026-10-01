#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import collections, hashlib, json, pathlib, urllib.request, zipfile
import numpy as np
root = pathlib.Path('/datasets/mavi-27-class')
root.mkdir(parents=True, exist_ok=True)
manifest = root/'manifest.json'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
if manifest.exists():
    d=json.loads(manifest.read_text())
    for name,meta in d['files'].items():
        assert sha(root/name)==meta['sha256'], name
    print(manifest.read_text()); raise SystemExit()
url='https://www.kaggle.com/api/v1/datasets/download/ardamavi/27-class-sign-language-dataset?datasetVersionNumber=1'
archive=root/'source-v1.zip'
if not archive.exists():
    urllib.request.urlretrieve(url, root/'source-v1.zip.part')
    (root/'source-v1.zip.part').rename(archive)
assert sha(archive)=='1f5bee3ee04209e2337d1cd7eeab73a0cc47809621558f4c14a7bf725a819704', 'Archive identity changed'
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())=={'X.npy','Y.npy'}, z.namelist()
    for name in ('X.npy','Y.npy'):
        if not (root/name).exists(): z.extract(name,root)
x=np.load(root/'X.npy',mmap_mode='r',allow_pickle=False)
y=np.load(root/'Y.npy',allow_pickle=False).reshape(-1)
assert x.shape==(22801,128,128,3) and x.dtype==np.float32
assert y.shape==(22801,) and len(set(y))==27
assert collections.Counter(y)['NULL']==314
assert float(x.min())>=0 and float(x.max())<=1
record={'source_url':url,'version':1,'license':'CC BY-NC-SA 4.0','research_permission':'Dataset card and Mavi & Dikle section 3 allow research use; IRB/volunteer consent in section 2.1. Noncommercial study processing only.','sample_count':len(y),'class_counts':dict(sorted(collections.Counter(y).items())),'shape':list(x.shape),'dtype':str(x.dtype),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [archive,root/'X.npy',root/'Y.npy']}}
manifest.write_text(json.dumps(record,indent=2)+'\n'); print(manifest.read_text())
PY
