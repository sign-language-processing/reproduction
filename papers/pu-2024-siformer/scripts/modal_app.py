"""Bounded real-data execution of pinned author code; use the repository Modal wrapper."""
from pathlib import Path
import modal

app = modal.App('d4719e6c-siformer')
image = (modal.Image.from_registry('ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291')
         .apt_install('git')
         .pip_install('pandas==2.2.3', 'scikit-learn==1.6.1', 'matplotlib==3.10.1', 'opencv-python-headless==4.11.0.86')
         .run_commands('git clone https://github.com/mpuu00001/Siformer /opt/siformer',
                       'git -C /opt/siformer checkout 979a14ed15ed0f20afd77d447ad23c4f4107a2c3')
         .add_local_file(Path(__file__).parent.parent / 'upstream.patch', '/opt/upstream.patch', copy=True)
         .run_commands('cd /opt/siformer && git apply /opt/upstream.patch')
         .env({'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub', 'MPLBACKEND': 'Agg'})
         .add_local_file(Path(__file__).with_name('preflight.py'), '/opt/preflight.py')
         .add_local_file(Path(__file__).with_name('rectification_check.py'), '/opt/rectification_check.py'))
datasets = modal.Volume.from_name('datasets', version=2)
cache = modal.Volume.from_name('huggingface-cache', version=2)
outputs = modal.Volume.from_name('d4719e6c-siformer-results', create_if_missing=True)

@app.function(image=image, gpu='A10G', cpu=4, memory=16000, timeout=1800,
              volumes={'/datasets': datasets.with_mount_options(read_only=True), '/cache/huggingface': cache, '/outputs': outputs})
def preflight(run_id: str):
    import os, subprocess, json, datetime
    out = Path('/outputs') / run_id
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Choose a fresh run ID; existing evidence is immutable.')
    os.chdir('/opt/siformer')
    meta = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'function_call_id': modal.current_function_call_id(), 'app_id': app.app_id,
            'command': 'python /opt/preflight.py /datasets/lsa64/siformer/LSA64_60fps.csv ' + str(out)}
    (out / 'pip-freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
    (out / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    with (out / 'stdout.log').open('w') as log:
        p = subprocess.run(['python','/opt/preflight.py','/datasets/lsa64/siformer/LSA64_60fps.csv',str(out)], stdout=log,stderr=subprocess.STDOUT)
    meta.update(exit_code=p.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'execution.json').write_text(json.dumps(meta,indent=2))
    outputs.commit()
    print(json.dumps(meta))
    print((out / 'stdout.log').read_text()[-12000:])
    return meta

@app.function(image=modal.Image.debian_slim(python_version='3.12').apt_install('curl')
              .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'})
              .add_local_file(Path(__file__).with_name('data.sh'), '/opt/data.sh'), timeout=2400,
              volumes={'/datasets':datasets, '/cache/huggingface':cache})
def acquire_data():
    import subprocess
    subprocess.run(['bash','/opt/data.sh'],check=True)
    datasets.commit()

@app.function(image=modal.Image.debian_slim(python_version='3.12')
              .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'}),
              cpu=2,memory=2048,timeout=900,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def fingerprint(run_id:str):
    import csv,io,urllib.request,itertools,ast,hashlib,json,datetime
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    result={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'app_id':app.app_id,'function_call_id':modal.current_function_call_id(),
            'original_asset_id':51405939,'original_asset_updated_at':'2021-12-09T16:57:04Z'}
    url='https://github.com/maty-bohacek/spoter/releases/download/supplementary-data/LSA64_60fps.csv'
    with urllib.request.urlopen(url,timeout=60) as response:
        reader=csv.DictReader(io.TextIOWrapper(response,encoding='utf-8'))
        original=list(itertools.islice(reader,16)); result['original_columns']=reader.fieldnames
    with Path('/datasets/lsa64/siformer/LSA64_60fps.csv').open() as f:
        reader=csv.DictReader(f);released=list(itertools.islice(reader,128));result['released_columns']=reader.fieldnames
    def canonical(row):
        return {k:ast.literal_eval(v) if v.startswith('[') else v for k,v in row.items() if not k.startswith('Unnamed')}
    original=[canonical(r) for r in original];released=[canonical(r) for r in released]
    result['original_sample_sha256']=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
    result['released_sample_sha256']=hashlib.sha256(json.dumps(released,sort_keys=True).encode()).hexdigest()
    comparisons=[]
    for i,a in enumerate(original):
        keys=[k for k,v in a.items() if isinstance(v,list) and k in released[0]]
        candidates=[]
        for n,b in enumerate(released):
            # Published SPOTER loader subtracts one; Siformer loader does not.
            if int(a['labels'])-1!=int(b['labels']):continue
            first=keys[0];av=a[first];bv=b[first]
            if av[:min(len(av),len(bv))] != bv[:min(len(av),len(bv))]:continue
            prefix=all(a[k][:min(len(a[k]),len(b[k]))]==b[k][:min(len(a[k]),len(b[k]))] for k in keys)
            candidates.append({'released_row':n,'all_coordinate_prefixes_equal':prefix,
                               'original_frames':len(av),'released_frames':len(bv),
                               'all_coordinate_lists_equal':all(a[k]==b[k] for k in keys),
                               'all_first_coordinates_equal':all(a[k][0]==b[k][0] for k in keys if a[k] and b[k])})
        comparisons.append({'original_row':i,'label':a.get('labels'),'coordinate_columns':len(keys),'matching_candidates':candidates,
                            'first_landmark_first_5':a[keys[0]][:5]})
    result['label_mapping']='SPOTER label minus1, as its published loader; Siformer label unchanged.'
    result['comparisons']=comparisons
    result['released_labels']=sorted(set(r['labels'] for r in released))
    result['released_first_landmark_first_5']=released[0][keys[0]][:5]
    result['sample_scope']='First16 original release records versus first128 Siformer records; no whole-corpus identity claim.'
    result['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'fingerprint.json').write_text(json.dumps(result,indent=2));outputs.commit();print(json.dumps(result))

@app.function(image=modal.Image.debian_slim(python_version='3.12')
              .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'}),
              cpu=2,memory=2048,timeout=900,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def full_fingerprint(run_id:str):
    import csv,io,urllib.request,itertools,ast,hashlib,json,datetime,collections
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    result={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'app_id':app.app_id,'function_call_id':modal.current_function_call_id(),
            'original_asset_id':51405939,'original_asset_updated_at':'2021-12-09T16:57:04Z'}
    h=hashlib.sha256()
    class HashReader(io.RawIOBase):
        def readable(self):return True
        def readinto(self,b):
            chunk=response.read(len(b));h.update(chunk);b[:len(chunk)]=chunk;return len(chunk)
    url='https://github.com/maty-bohacek/spoter/releases/download/supplementary-data/LSA64_60fps.csv'
    counts=collections.Counter();padding=collections.Counter();lengths=collections.Counter();mismatches=[]
    with urllib.request.urlopen(url,timeout=60) as response, Path('/datasets/lsa64/siformer/LSA64_60fps.csv').open() as released:
        original=csv.DictReader(io.TextIOWrapper(io.BufferedReader(HashReader()),encoding='utf-8'))
        target=csv.DictReader(released)
        for i,(a,b) in enumerate(itertools.zip_longest(original,target)):
            if a is not None:counts['original_records']+=1
            if b is not None:counts['released_records']+=1
            if a is None or b is None:counts['unpaired_records']+=1;continue
            counts['paired_records']+=1
            if int(a['labels'])-1==int(b['labels']):counts['mapped_labels_equal']+=1
            row_equal=True;original_length=None;released_length=None
            for key,value in a.items():
                if not value.startswith('[') or key not in b:continue
                x=ast.literal_eval(value);y=ast.literal_eval(b[key]);original_length=len(x);released_length=len(y)
                counts['coordinate_columns_compared']+=1
                same=len(y)>=len(x) and x==y[:len(x)]
                if same:counts['coordinate_prefixes_equal']+=1
                else:
                    row_equal=False
                    if len(mismatches)<10:mismatches.append({'row':i,'column':key,'original_frames':len(x),'released_frames':len(y)})
                tail=y[len(x):]
                if not tail:padding['none']+=1
                elif all(v==0 for v in tail):padding['zero']+=1
                elif x and all(v==x[-1] for v in tail):padding['repeat_last']+=1
                else:padding['other']+=1
            if row_equal:counts['all_original_coordinates_preserved_records']+=1
            lengths[(original_length,released_length)]+=1
    result.update(counts=dict(counts),padding_columns=dict(padding),length_pairs=[{'original_frames':a,'released_frames':b,'records':n} for (a,b),n in sorted(lengths.items())],mismatch_examples=mismatches,original_sha256=h.hexdigest(),label_mapping='Original published SPOTER loader subtracts1; Siformer loader leaves released zero-based labels unchanged.',method='Pair all CSV records in released order, parse list-valued coordinates exactly, compare every original coordinate to released prefix; classify appended values. No normalization or rounding.',finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'fingerprint.json').write_text(json.dumps(result,indent=2));outputs.commit();print(json.dumps(result))

@app.function(image=image,cpu=4,memory=4096,timeout=900,
              volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/outputs':outputs})
def rectification_check(run_id:str):
    import subprocess,json,datetime,hashlib
    out=Path('/outputs')/run_id;out.mkdir(parents=True,exist_ok=False)
    result={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'app_id':app.app_id,'function_call_id':modal.current_function_call_id()}
    (out/'freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
    with (out/'stdout.log').open('w') as log:
        p=subprocess.run(['python','/opt/rectification_check.py',str(out)],stdout=log,stderr=subprocess.STDOUT)
    result.update(exit_code=p.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'execution.json').write_text(json.dumps(result,indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2));outputs.commit();print(json.dumps(result))
    return result

@app.local_entrypoint()
def main(run_id: str = 'preflight-001', acquire: bool=False, compare_features: bool=False, compare_all: bool=False, check_rectification: bool=False):
    if check_rectification:
        result=rectification_check.remote(run_id)
        if result['exit_code']:raise SystemExit(result['exit_code'])
        return
    if compare_all:
        full_fingerprint.remote(run_id);return
    if compare_features:
        fingerprint.remote(run_id);return
    print(acquire_data.remote() if acquire else preflight.remote(run_id))
