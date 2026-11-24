"""Adapt the public per-sentence features to the cited author's existing loader."""
from pathlib import Path
import csv, gzip, hashlib, json, pickle, sys, time, zipfile, tempfile
import numpy as np
import torch

root=Path('/datasets/how2sign/spot-align-wicv2023')
out=root/'slt-format';out.mkdir(exist_ok=True)
source_manifest=json.loads((root/'manifest.json').read_text())
source_pin=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
files_pin=hashlib.sha256(json.dumps(source_manifest['files'],sort_keys=True).encode()).hexdigest()
script_pin=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest=out/'manifest.json'
if manifest.exists():
    old=json.loads(manifest.read_text())
    if old['source_files_sha256']==files_pin and old['adapter_sha256']==script_pin:
        print(json.dumps(old));sys.exit(0)
    raise ValueError('Existing derived data has a different source or adapter; preserve it.')
result={'source_manifest_sha256':source_pin,'source_files_sha256':files_pin,'adapter_sha256':script_pin,'splits':{},'started_unix':time.time()}
for split in ['train','val','test']:
    with (root/(split+'.tsv')).open() as f:
        rows=list(csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE))
    # Copy the single pinned archive to ephemeral disk to avoid tens of
    # thousands of small-file reads from the shared Volume. Inputs stay intact.
    local_zip=Path(tempfile.gettempdir())/('cabot-'+split+'.zip')
    copy_hash=hashlib.sha256();copy_started=time.time()
    with (root/(split+'.zip')).open('rb') as src,local_zip.open('wb') as dst:
        for chunk in iter(lambda:src.read(8*1024*1024),b''):
            dst.write(chunk);copy_hash.update(chunk)
    expected=next(x['sha256'] for x in source_manifest['files'] if x['name']==split+'.zip')
    assert copy_hash.hexdigest()==expected,'Copied archive hash mismatch'
    archive=zipfile.ZipFile(local_zip)
    members=[i.filename for i in archive.infolist() if i.filename.endswith('.npy') and '__MACOSX' not in i.filename]
    feature_paths={Path(name).stem:name for name in members}
    print(json.dumps({'split':split,'archive_copy_seconds':time.time()-copy_started,'archive_sha256':expected}),flush=True)
    assert len(feature_paths)==len(members),'Duplicate feature IDs in archive'
    samples=[];removed=[];parity_count=0
    for row in rows:
        # The released NPY is already the sentence slice. The TSV offset refers
        # to its source video and must not be applied to this NPY a second time.
        path=feature_paths.get(row['id'])
        if path is None:
            removed.append({'id':row['id'],'reason':'missing_from_published_feature_archive'});continue
        with archive.open(path) as member:
            features=np.load(member,allow_pickle=False)
        if parity_count<16:
            reference=np.load(root/path,allow_pickle=False)
            assert np.array_equal(features,reference),'Archive/extracted array mismatch'
            parity_count+=1
            if parity_count==16:print(json.dumps({'split':split,'archive_vs_extracted_parity_records':parity_count,'all_equal':True}),flush=True)
        if features.ndim!=2 or features.shape[1]!=1024:
            raise ValueError('Unexpected feature shape: '+row['id'])
        if features.shape[0]==0:
            removed.append({'id':row['id'],'reason':'empty_feature_sequence'});continue
        if not np.isfinite(features).all():raise ValueError('Nonfinite features: '+row['id'])
        samples.append({'name':row['id'],'signer':row['signer_id'],'gloss':row['translation'],
                        'text':row['translation'],'sign':torch.from_numpy(features)})
    archive.close();local_zip.unlink()
    assert sum(r['reason']=='missing_from_published_feature_archive' for r in removed)==(4 if split=='train' else 0),split
    target=out/(split+'.pkl.gz');temporary=out/(split+'.pkl.gz.partial')
    # Keep ordering, gzip metadata and pickle protocol fixed; record actual bytes.
    with temporary.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as f:
        pickle.dump(samples,f,protocol=4)
    temporary.rename(target)
    h=hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    lengths=[len(s['sign']) for s in samples]
    record={'archive_vs_extracted_parity_records':parity_count,'input_rows':len(rows),'retained_rows':len(samples),'removed':removed,
            'native_training_filter_eligible':sum(len(s['sign'])<=400 and len(s['text'].split())<=400 for s in samples),
            'max_timesteps':max(lengths),'sha256':h.hexdigest(),'size_bytes':target.stat().st_size}
    if split in ['val','test']:
        ordered=sorted(samples,key=lambda x:len(x['sign']))
        indices=np.linspace(0,len(ordered)-1,16,dtype=int)
        diagnostic=[ordered[int(i)] for i in indices]
        diagnostic_path=out/(split+'-preflight.pkl.gz')
        with diagnostic_path.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as f:
            pickle.dump(diagnostic,f,protocol=4)
        record['diagnostic']={'path':diagnostic_path.name,'count':len(diagnostic),'selection':'16 evenly spaced length ranks, including shortest/longest; diagnostics only','sha256':hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()}
        del ordered,diagnostic
    result['splits'][split]=record;print(json.dumps({split:record}),flush=True)
    del samples
result['finished_unix']=time.time();manifest.write_text(json.dumps(result,indent=2));print(json.dumps(result))
