"""Cabot Table 5 reproduction using the cited neccam/slt recipe and released features."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[3] if modal.is_local() else Path('/workspace')
# Reuse the repository's reviewed legacy environment; do not duplicate author code.
LEGACY=ROOT/'papers/camgoz-2020-slt'
base_image=modal.Image.from_dockerfile(LEGACY/'Dockerfile',context_dir=LEGACY,add_python='3.11')
image=(base_image
       .add_local_file(Path(__file__).with_name('run.py'),'/opt/cabot-run.py')
       .add_local_file(Path(__file__).with_name('collect.py'),'/opt/collect.py')
       .add_local_file(Path(__file__).with_name('prepare_how2.py'),'/opt/prepare_how2.py'))
how2_image=(base_image
       .add_local_dir(Path(__file__).resolve().parents[1]/'patches','/opt/how2-patches',copy=True)
       .run_commands('git -C /slt apply /opt/how2-patches/01-english-wer.patch', 'git -C /slt apply /opt/how2-patches/02-wer-integer-range.patch', 'git -C /slt apply /opt/how2-patches/03-training-recovery.patch')
       .add_local_file(Path(__file__).with_name('run.py'),'/opt/cabot-run.py')
       .add_local_file(Path(__file__).with_name('collect.py'),'/opt/collect.py')
       .add_local_file(Path(__file__).with_name('evaluation.py'),'/opt/evaluation.py')
       .add_local_file(Path(__file__).with_name('check_evaluation.py'),'/opt/check_evaluation.py')
       .add_local_file(Path(__file__).with_name('parallel_ctc.py'),'/opt/parallel_ctc.py')
       .add_local_file(Path(__file__).with_name('recovery_train.py'),'/opt/recovery_train.py')
       .add_local_file(Path(__file__).with_name('check_training_recovery.py'),'/opt/check_training_recovery.py'))
app=modal.App('7f7abc3e-cabot-how2sign')
datasets=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
outputs=modal.Volume.from_name('7f7abc3e-cabot-results',create_if_missing=True,version=2)
PYTHON='/root/miniconda3/bin/python'

def run_experiment(mode:str,run_id:str):
    import subprocess,os,datetime,json,threading,hashlib
    out=Path('/outputs')/run_id
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Choose a new run ID to preserve evidence.')
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None)
    meta=dict(started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),app_id=app.app_id,function_call_id=modal.current_function_call_id(),mode=mode,command=PYTHON+' /opt/cabot-run.py '+mode+' '+str(out))
    (out/'execution.json').write_text(json.dumps(meta,indent=2))
    source_dir=out/'entrypoint-source';source_dir.mkdir()
    source_hashes={}
    for name in ['cabot-run.py','collect.py','evaluation.py','parallel_ctc.py']:
        path=Path('/opt')/name
        if path.exists():
            content=path.read_bytes();(source_dir/name).write_bytes(content)
            source_hashes[name]=hashlib.sha256(content).hexdigest()
    (source_dir/'sha256.json').write_text(json.dumps(source_hashes,indent=2))
    (out/'upstream-diff.patch').write_text(subprocess.check_output(['git','-C','/slt','diff'],text=True))
    (out/'freeze.txt').write_text(subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True,env=env))
    (out/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    is_how2=mode.startswith('how2-')
    if is_how2:
        manifest_path=Path('/datasets/how2sign/spot-align-wicv2023/slt-format/manifest.json')
        manifest=json.loads(manifest_path.read_text())
        meta['data_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        paths=[(manifest_path.parent/(split+'.pkl.gz'),record['sha256']) for split,record in manifest['splits'].items()]
        paths.extend((manifest_path.parent/record['diagnostic']['path'],record['diagnostic']['sha256']) for record in manifest['splits'].values() if 'diagnostic' in record)
    else:
        expected={'train':'196842893dd43c98a5574132dceccf50f3e8f95af853042377cf05519757a773','dev':'5be78d8488eaa4400e3bfcac1cf9096f7932595e4b5116eab53c19024693c6e6','test':'068c85c2f675e21ce0e6e4e9d419bc63fac7f43678783a7e5eb452ecb38c566a'}
        paths=[(Path('/datasets/rwth-phoenix-2014-t/features/author/PHOENIX2014T')/('phoenix14t.pami0.'+split),digest) for split,digest in expected.items()]
    for path,digest in paths:
        with path.open('rb') as f:
            h=hashlib.sha256()
            while chunk:=f.read(8*1024*1024): h.update(chunk)
        assert h.hexdigest()==digest,str(path)
    stop=threading.Event()
    def commit():
        while not stop.wait(120):outputs.commit()
    worker=threading.Thread(target=commit,daemon=True);worker.start()
    try:
        with (out/'stdout.log').open('w') as log:
            p=subprocess.run([PYTHON,'/opt/cabot-run.py',mode,str(out)],cwd='/slt',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800 if mode.endswith('preflight') else (85000 if is_how2 else 21000))
        meta['exit_code']=p.returncode
    except subprocess.TimeoutExpired:
        meta['exit_code']=124
    finally:
        stop.set();worker.join()
        meta['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/'execution.json').write_text(json.dumps(meta,indent=2));outputs.commit()
    print(json.dumps(meta));print((out/'stdout.log').read_text()[-10000:])
    return meta

@app.function(image=image,gpu='T4',cpu=4,memory=32768,timeout=21600,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def execute(mode:str,run_id:str):
    return run_experiment(mode,run_id)

@app.function(image=how2_image,gpu='T4',cpu=4,memory=65536,timeout=86400,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def execute_how2(mode:str,run_id:str):
    return run_experiment('how2-'+mode,run_id)

@app.function(image=modal.Image.debian_slim(python_version='3.12').add_local_file(Path(__file__).with_name('data.sh'),'/opt/data.sh'),timeout=10800,volumes={'/datasets':datasets,'/cache/huggingface':cache},secrets=[])
def acquire_data():
    import os, subprocess
    os.environ.update(HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub')
    subprocess.run(['bash','/opt/data.sh'],check=True)
    datasets.commit()

@app.function(image=modal.Image.debian_slim(python_version='3.12').add_local_file(Path(__file__).with_name('data.sh'),'/opt/data.sh'),
              cpu=2,memory=2048,timeout=900,volumes={'/datasets':datasets,'/cache/huggingface':cache,'/outputs':outputs})
def finish_acquisition(run_id:str):
    import os,subprocess,json,datetime
    os.environ.update(HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub')
    root=Path('/datasets/how2sign/spot-align-wicv2023')
    for name in ['Readme.txt','train.tsv','val.tsv','test.tsv','train.zip','val.zip','test.zip']:
        assert (root/name).is_file(),name
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    with (out/'stdout.log').open('w') as log:
        p=subprocess.run(['bash','/opt/data.sh'],stdout=log,stderr=subprocess.STDOUT,timeout=800)
    meta.update(exit_code=p.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'execution.json').write_text(json.dumps(meta,indent=2));datasets.commit();outputs.commit();print(json.dumps(meta));print((out/'stdout.log').read_text())
    if p.returncode:raise RuntimeError('Finalization failed')

@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=1,memory=1024,timeout=1800,
              volumes={'/datasets':datasets,'/cache/huggingface':cache,'/outputs':outputs})
def acquire_validation(run_id:str):
    """Populate the independent validation archive without sharing a partial writer."""
    import urllib.request,hashlib,json,datetime,os,zipfile
    os.environ.update(HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub')
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    root=Path('/datasets/how2sign/spot-align-wicv2023')
    staged=root/('val.zip.'+run_id+'.staged')
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'app_id':app.app_id,'function_call_id':modal.current_function_call_id(),
          'url':'https://dataverse.csuc.cat/api/access/datafile/51922'}
    print(json.dumps(meta),flush=True)
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(meta['url'],timeout=180) as response,staged.open('xb') as f:
        while chunk:=response.read(8*1024*1024):
            f.write(chunk);h.update(chunk);size+=len(chunk)
    with zipfile.ZipFile(staged) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failure: '+bad)
        meta['npy_count']=sum(x.filename.endswith('.npy') and '__MACOSX' not in x.filename for x in z.infolist())
    target=root/'val.zip';train=root/'train.zip.partial'
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==h.hexdigest()
        meta['publication']='Identical final archive already present; staged copy retained.'
    elif train.exists() and 7778524363-train.stat().st_size>=256*1024*1024 and not (root/'val.zip.partial').exists():
        # At least 256 MiB of serial training download remains. Publish completed
        # bytes atomically; the existing loop will then skip validation download.
        staged.rename(target)
        meta['publication']='Atomic final archive published while serial training download remained active.'
    else:
        meta['publication']='Main downloader too close to validation; separate staged copy retained without modifying its paths.'
    meta.update(bytes=size,sha256=h.hexdigest(),finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=0)
    (out/'acquisition.json').write_text(json.dumps(meta,indent=2));datasets.commit();outputs.commit();print(json.dumps(meta))

@app.function(image=modal.Image.debian_slim(python_version='3.12'),timeout=60,volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache})
def inspect_data():
    import zipfile,csv,struct,ast
    root=Path('/datasets/how2sign/spot-align-wicv2023')
    print([(p.name,p.stat().st_size) for p in root.iterdir()])
    with zipfile.ZipFile(root/'test.zip') as z:
        print(z.namelist()[:10])
        for name in [n for n in z.namelist() if n.endswith('.npy') and '__MACOSX' not in n][:3]:
            with z.open(name) as f:
                magic=f.read(8); length=struct.unpack('<H',f.read(2))[0]; print(name,ast.literal_eval(f.read(length).decode('latin1')))
    for p in root.glob('*.tsv'):
        with p.open() as f:print(p.name,len(list(csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE))))

@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=2,memory=2048,timeout=1800,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def audit_data(run_id:str):
    import zipfile,csv,struct,ast,json,datetime,os,hashlib
    os.environ.update(HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub')
    root=Path('/datasets/how2sign/spot-align-wicv2023')
    out=Path('/outputs')/run_id
    out.mkdir(parents=True,exist_ok=False)
    result={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'app_id':app.app_id,'function_call_id':modal.current_function_call_id(),'splits':{}}
    for split in ['train','val','test']:
        shapes=[]; names=[]; dtypes={}
        with zipfile.ZipFile(root/(split+'.zip')) as z:
            for info in z.infolist():
                if not info.filename.endswith('.npy') or '__MACOSX' in info.filename:continue
                with z.open(info) as f:
                    magic=f.read(8)
                    if magic[:6]!=b'\x93NUMPY':raise ValueError('Not a NumPy file')
                    size=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0]
                    header=ast.literal_eval(f.read(size).decode('latin1'))
                shapes.append(header['shape']);names.append(Path(info.filename).stem)
                dtypes[header['descr']]=dtypes.get(header['descr'],0)+1
        with (root/(split+'.tsv')).open() as f:
            rows=list(csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE))
        shape_by_name=dict(zip(names,shapes))
        eligible=sum(r['id'] in shape_by_name and shape_by_name[r['id']][0]>0 and (split!='train' or (shape_by_name[r['id']][0]<=400 and len(r['translation'].split())<=400)) for r in rows)
        result['splits'][split]={'nonempty_and_native_filter_eligible_rows':eligible,
            'duplicate_tsv_ids':len(rows)-len(set(r['id'] for r in rows)),
            'archive_npy_count':len(shapes),'tsv_row_count':len(rows),'dtype_counts':dtypes,
            'tsv_ids_missing_features':len(set(r['id'] for r in rows)-set(names)),
            'feature_ids_missing_tsv':len(set(names)-set(r['id'] for r in rows)),
            'over_400_text_tokens_count':sum(len(r['translation'].split())>400 for r in rows),
            'zero_dimension_count':sum(any(x==0 for x in shape) for shape in shapes),
            'non_1024_channel_count':sum(len(shape)!=2 or shape[-1]!=1024 for shape in shapes),
            'min_timesteps':min(shape[0] for shape in shapes),'max_timesteps':max(shape[0] for shape in shapes),
            'over_400_timesteps_count':sum(shape[0]>400 for shape in shapes),
            'sorted_feature_names_sha256':hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest()}
    result['nonempty_and_native_filter_eligible_total']=sum(x['nonempty_and_native_filter_eligible_rows'] for x in result['splits'].values())
    result['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'audit.json').write_text(json.dumps(result,indent=2));outputs.commit();print(json.dumps(result))

@app.function(image=image,cpu=4,memory=65536,timeout=3600,
              volumes={'/datasets':datasets,'/cache/huggingface':cache,'/outputs':outputs})
def prepare_how2(run_id:str):
    import subprocess,os,json,datetime,shutil
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    with (out/'stdout.log').open('w') as log:
        p=subprocess.run([PYTHON,'/opt/prepare_how2.py'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3500)
    meta.update(exit_code=p.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if p.returncode==0:
        root=Path('/datasets/how2sign/spot-align-wicv2023')
        shutil.copy2(root/'manifest.json',out/'source-manifest.json')
        shutil.copy2(root/'slt-format/manifest.json',out/'derived-manifest.json')
    (out/'execution.json').write_text(json.dumps(meta,indent=2));datasets.commit();outputs.commit()
    print(json.dumps(meta));print((out/'stdout.log').read_text()[-20000:])
    if p.returncode:raise RuntimeError('Data adapter failed: '+str(p.returncode))

@app.function(image=image,cpu=2,memory=4096,timeout=600,volumes={'/outputs':outputs,'/cache/huggingface':cache})
def collect(run_id:str):
    import subprocess,os
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    subprocess.run([PYTHON,'/opt/collect.py',str(Path('/outputs')/run_id)],env=env,check=True)
    outputs.commit()


@app.function(image=how2_image,cpu=1,memory=4096,timeout=300,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def check_how2_metric(run_id:str):
    import subprocess,os,json,datetime
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    code="""
import csv,json,subprocess,numpy as np
from pathlib import Path
from signjoey import metrics as fixed
old={};exec(subprocess.check_output(['git','show','HEAD:signjoey/metrics.py']).decode(),old)
rows=list(csv.DictReader(open('/datasets/how2sign/spot-align-wicv2023/test.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE))
selected=sorted(rows,key=lambda r:len(r['translation'].split()),reverse=True)[:2]
results=[]
for row in selected:
 r=row['translation'];h=' '.join(r.split()[:30]);a=old['wer_single'](r,h);b=fixed.wer_single(r,h)
 assert b['num_del']==len(r.split())-30 and b['num_ins']==0 and b['num_sub']==0
 results.append({'id':row['id'],'reference_words':len(r.split()),'diagnostic_hypothesis':'First30 reference tokens; no model output','legacy_errors':int(a['num_err']),'widened_errors':int(b['num_err']),'expected_deletions':len(r.split())-30})
assert any(r['legacy_errors']!=r['widened_errors'] for r in results)
print(json.dumps({'numpy_version':np.__version__,'source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'fixtures':results},indent=2))
"""
    p=subprocess.run([PYTHON,'-c',code],cwd='/slt',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
    (out/'stdout.log').write_text(p.stdout);meta.update(exit_code=p.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'execution.json').write_text(json.dumps(meta,indent=2));outputs.commit();print(json.dumps(meta));print(p.stdout)
    if p.returncode:raise RuntimeError('Metric diagnostic failed')


@app.function(image=image,cpu=1,memory=4096,timeout=300,
              volumes={'/outputs':outputs,'/cache/huggingface':cache})
def check_phoenix_metric(run_id:str):
    import subprocess,os,json,datetime
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    code="""
import sys,io,json,pickle,torch
from pathlib import Path
sys.path.insert(0,'/slt')
from signjoey import metrics
safe={};exec(Path('/slt/signjoey/metrics.py').read_text().replace('dtype=np.uint8','dtype=np.int64'),safe)
class CPUUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='torch.storage' and name=='_load_from_bytes':return lambda b:torch.load(io.BytesIO(b),map_location='cpu')
  return super().find_class(module,name)
root=Path('/outputs/phoenix-full-seed42-001');result={}
for split in ['dev','test']:
 path=next(root.rglob('*.'+split+'_results.pkl'))
 with path.open('rb') as f:d=CPUUnpickler(f).load()
 if split=='dev':d=min(d['recognition_results'].items(),key=lambda x:x[1]['valid_scores']['wer'])[1]
 refs,hyps=d['gls_ref'],d['gls_hyp']
 legacy=metrics.wer_list(refs,hyps);wide=safe['wer_list'](refs,hyps)
 assert legacy==wide
 result[split]={'records':len(refs),'reference_max_words':max(map(lambda x:len(x.split()),refs)),'hypothesis_max_words':max(map(lambda x:len(x.split()),hyps)),'legacy_uint8':legacy,'widened_int64':wide,'identical':True}
print(json.dumps(result,indent=2))
"""
    q=subprocess.run([PYTHON,'-c',code],cwd='/slt',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
    (out/'stdout.log').write_text(q.stdout);meta.update(exit_code=q.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'execution.json').write_text(json.dumps(meta,indent=2));outputs.commit();print(json.dumps(meta));print(q.stdout)
    if q.returncode:raise RuntimeError('PHX metric audit failed')


@app.function(image=how2_image,cpu=8,memory=32768,timeout=3600,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def check_evaluation(run_id:str,variant:str="cpu"):
    import subprocess,os,json,datetime
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    try:
        with (out/'stdout.log').open('w') as log:
            q=subprocess.run([PYTHON,'/opt/check_evaluation.py',str(out),variant],cwd='/slt',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3500)
        meta['exit_code']=q.returncode
    except subprocess.TimeoutExpired:meta['exit_code']=124
    meta['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'execution.json').write_text(json.dumps(meta,indent=2));outputs.commit();print(json.dumps(meta));print((out/'stdout.log').read_text()[-10000:])
    if meta['exit_code']:raise RuntimeError('Equivalence diagnostic failed')


@app.function(image=how2_image,gpu='T4',cpu=4,memory=32768,timeout=1200,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def check_evaluation_gpu(run_id:str,variant:str="gpu"):
    import subprocess,os,json,datetime
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    (out/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    try:
        with (out/'stdout.log').open('w') as log:
            q=subprocess.run([PYTHON,'/opt/check_evaluation.py',str(out),variant],cwd='/slt',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1100)
        meta['exit_code']=q.returncode
    except subprocess.TimeoutExpired:meta['exit_code']=124
    meta['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'execution.json').write_text(json.dumps(meta,indent=2));outputs.commit();print(json.dumps(meta));print((out/'stdout.log').read_text()[-10000:])
    if meta['exit_code']:raise RuntimeError('GPU equivalence diagnostic failed')

@app.function(image=how2_image,cpu=4,memory=32768,timeout=1200,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def check_training_recovery(run_id:str,inspect_run_id:str=""):
    import subprocess,os,json,datetime
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HOME='/cache/huggingface',HF_HUB_CACHE='/cache/huggingface/hub',PYTHONNOUSERSITE='1',PYTHONPATH='/opt:/slt')
    meta={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    try:
        with (out/'stdout.log').open('w') as log:
            q=subprocess.run([PYTHON,'/opt/check_training_recovery.py',str(Path('/outputs')/inspect_run_id),'inspect'] if inspect_run_id else [PYTHON,'/opt/check_training_recovery.py',str(out)],cwd='/slt',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1100)
        meta['exit_code']=q.returncode
    except subprocess.TimeoutExpired:meta['exit_code']=124
    meta['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'execution.json').write_text(json.dumps(meta,indent=2));outputs.commit();print(json.dumps(meta));print((out/'stdout.log').read_text()[-10000:])
    if meta['exit_code']:raise RuntimeError('Training recovery diagnostic failed')

@app.local_entrypoint()
def main(mode:str='preflight',run_id:str='preflight-001',source_run_id:str=''):
    if mode=='inspect-training-recovery':
        check_training_recovery.remote(run_id,source_run_id); return
    if mode=='check-training-recovery':
        check_training_recovery.remote(run_id); return
    if mode=='check-capacity-normal':
        check_evaluation_gpu.remote(run_id,'capacity-normal'); return
    if mode=='check-capacity':
        check_evaluation_gpu.remote(run_id,'capacity'); return
    if mode=='check-evaluation-gpu':
        check_evaluation_gpu.remote(run_id); return
    if mode=='check-evaluation-remaining':
        check_evaluation.remote(run_id,'remaining'); return
    if mode=='check-evaluation':
        check_evaluation.remote(run_id); return
    if mode=='check-phoenix-metric':
        check_phoenix_metric.remote(run_id); return
    if mode=='check-how2-metric':
        check_how2_metric.remote(run_id); return
    if mode=='finish-acquisition':
        finish_acquisition.remote(run_id); return
    if mode=='prepare-how2':
        prepare_how2.remote(run_id); return
    if mode=='acquire-validation':
        acquire_validation.remote(run_id); return
    if mode=='audit-data':
        audit_data.remote(run_id); return
    if mode=='collect':
        collect.remote(run_id); return
    if mode=='inspect-data':
        inspect_data.remote(); return
    if mode=='acquire':
        acquire_data.remote(); return
    if mode in ['how2-preflight','how2-full']:
        meta=execute_how2.remote(mode[len('how2-'):],run_id)
        print(meta)
        if meta['exit_code']:raise SystemExit(meta['exit_code'])
        return
    if mode not in ['preflight','full']:raise ValueError(mode)
    meta=execute.remote(mode,run_id)
    print(meta)
    if meta['exit_code']:
        raise SystemExit(meta['exit_code'])
