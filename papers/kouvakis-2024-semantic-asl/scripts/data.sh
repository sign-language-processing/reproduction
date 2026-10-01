#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import pathlib, urllib.request, hashlib, zipfile, json, collections
for slug,url,md5,expected_sha256,license in [
 ('asl-semcom-rgb','https://zenodo.org/records/14635573/files/ASL_SemCom.zip?download=1','831ff816c3bb36ffc3b0c9f248cf5033','686e818063a1a81bb90fb2cbb319f8c9dc8d368650faf71351d836e508ba0582','CC BY 4.0'),
 ('sign-language-mnist','https://www.kaggle.com/api/v1/datasets/download/datamunge/sign-language-mnist?datasetVersionNumber=1',None,'fa1b513570d4348c6d6860e04e5854c59ef8eadb8c42a45d36bd1286ce3d489f','CC0')]:
 root=pathlib.Path('/datasets')/slug
 if (root/'manifest.json').exists():
  assert json.loads((root/'manifest.json').read_text())['source_sha256']==expected_sha256
  print(json.dumps({'slug':slug,'status':'already populated','source_sha256':expected_sha256}));continue
 root.mkdir(parents=True,exist_ok=True)
 archive=root/'source.zip'
 urllib.request.urlretrieve(url,archive)
 raw=archive.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==expected_sha256
 if md5: assert hashlib.md5(raw).hexdigest()==md5
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   assert not name.startswith('/') and '..' not in pathlib.PurePosixPath(name).parts
  z.extractall(root/'files')
 counts=collections.Counter();files=[]
 for p in sorted((root/'files').rglob('*')):
  if p.is_file():
   files.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size})
   if p.suffix.lower() in ['.png','.jpg','.jpeg']:counts[str(p.parent.relative_to(root/'files'))]+=1
 manifest={'source':url,'license':license,'source_sha256':hashlib.sha256(raw).hexdigest(),'source_md5':hashlib.md5(raw).hexdigest(),'files':files,'image_counts':dict(counts)}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'slug':slug,'source_sha256':manifest['source_sha256'],'file_count':len(files),'image_counts':dict(counts),'nonimage_files':[x for x in files if not x['path'].lower().endswith(('.jpg','.png','.jpeg'))]}))
PY
