#!/usr/bin/env bash
set -euo pipefail
root=/datasets/how2sign/spot-align-wicv2023
mkdir -p "$root"
python - <<'PY'
from pathlib import Path
import urllib.request, hashlib, json, zipfile, datetime, shutil
root=Path('/datasets/how2sign/spot-align-wicv2023').resolve()
files={'Readme.txt':53890,'test.tsv':53222,'train.tsv':51923,'val.tsv':51537,'test.zip':51538,'train.zip':51543,'val.zip':51922}
expected={'Readme.txt': '39082fe3109068ab29bb5a6fb0fe9370ceb48c32341c3951aeeaaed2f563fd31', 'test.tsv': 'd404cd64997f443cc6c05154511da9571cd1b897ef24500b75f5a96568ef647b', 'train.tsv': '9e117ae049b83de9582bf39846d4094694f376fb59f4a0e5ad8f4bd3fe4ba65c', 'val.tsv': '3fd2a7006b6a54b9843dd9649b60010a51957364095ed128d42ce818021349b9', 'test.zip': '60c595b721649862d20cbc0ebccfa2b360fc87e5f564f2617f9e382d85620819', 'train.zip': '637ffbf99f7420d41a0b6ef7e5790292e5f32ec188cb68342d45485c5a377e1d', 'val.zip': '492e2807f1473ed0a17acb9b0365634fac1f10f84805e963ef6cd5c24dfea92f'}
records=[]
for name, file_id in files.items():
    path=root/name;url=f'https://dataverse.csuc.cat/api/access/datafile/{file_id}'
    if name.endswith('.tsv'):url+='?format=original'
    if not path.exists():
        partial=Path(str(path)+'.partial')
        offset=partial.stat().st_size if partial.exists() else 0
        print({'downloading':name,'resume_offset_bytes':offset},flush=True)
        req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(req,timeout=180) as response:
            resume=offset and response.status==206
            if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                raise ValueError('Unexpected range response')
            with partial.open('ab' if resume else 'wb') as f:shutil.copyfileobj(response,f,8*1024*1024)
        partial.rename(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==expected[name],name+' checksum mismatch'
    if name.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                target=(root/info.filename).resolve()
                if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
            marker=root/('.'+name+'.extracted')
            if not marker.exists():
                z.extractall(root)
                marker.write_text(str(path.stat().st_size))
            count=sum(i.filename.endswith('.npy') for i in z.infolist())
    else:count=None
    records.append(dict(name=name,url=url,bytes=path.stat().st_size,sha256=h.hexdigest(),npy_count=count))
    print(records[-1],flush=True)
(root/'manifest.json').write_text(json.dumps(dict(dataset='How2Sign SPOT-ALIGN features distributed with WiCV2023',doi='10.34810/DATA693',version='1.0',license='CC BY-NC-ND 4.0',accessed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records),indent=2))
PY
