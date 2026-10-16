"""CPU-only terminal evidence hashing; never changes GPU execution or native metrics."""
from pathlib import Path
import modal

APP = modal.App('repro-992e7a-faster-collect')
OUTPUT = modal.Volume.from_name('repro-992e7a-results', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
IMAGE = (modal.Image.debian_slim(python_version='3.12')
         .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub'})
         .add_local_file(Path(__file__), '/work/faster_collect.py'))


@APP.function(image=IMAGE, cpu=2, memory=4096, timeout=900, retries=0,
              volumes={'/outputs':OUTPUT,'/cache/huggingface':CACHE})
def collect(run_id: str):
    import datetime
    import hashlib
    import json
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor
    output = Path('/outputs')/run_id
    execution = json.loads((output/'execution.json').read_text())
    assert execution.get('exit_code') is not None, 'Collect only after terminal GPU execution.'
    assert all(s.get('state') != 'running' for s in execution['segments'])
    start = datetime.datetime.now(datetime.timezone.utc)
    clock = time.monotonic()
    receipt = {'started_at_utc':start.isoformat(),'modal_app_id':APP.app_id,
               'modal_function_call_id':modal.current_function_call_id(),
               'gpu_count':0,'cpu':2,'memory_mib':4096,'timeout_seconds':900,
               'collector_sha256':hashlib.sha256(Path('/work/faster_collect.py').read_bytes()).hexdigest()}
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        return {'path':str(path.relative_to(output)),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        return value
    code=1
    try:
        if execution['exit_code']==0:
            result=json.loads((output/'result.json').read_text())
            selected=result['selected']
            assert digest(output/selected['checkpoint'])['sha256']==selected['sha256']
            assert result['completed_updates']==30180 and result['epochs']==60
            raw={'native_result':'result.json','selected':selected,
                 'bbox':finite(result['test']['bbox']),
                 'all_parameters':result['all_parameters'],'million_parameters':result['all_parameters']/1e6,
                 'mean_inference_seconds':result['timing']['mean_seconds'],'fps':result['timing']['fps'],
                 'note':'Nonfinite native per-category/area fields become null here; original result.json is unchanged.'}
            (output/'collected-metrics.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
        paths=sorted(p for p in output.rglob('*') if p.is_file()
                     and p.name != 'evidence.json' and not p.name.startswith('collection-'))
        print(f'Hashing {len(paths)} terminal files with four bounded reader threads.',flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(digest,paths))
        (output/'evidence.json').write_text(json.dumps(files,indent=2)+'\n')
        receipt['file_count']=len(files)
        receipt['manifest_sha256']=hashlib.sha256((output/'evidence.json').read_bytes()).hexdigest()
        code=0
    finally:
        receipt.update(exit_code=code,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       wall_seconds=time.monotonic()-clock)
        (output/('collection-'+start.strftime('%Y%m%dT%H%M%S')+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        OUTPUT.commit()
    return receipt


@APP.local_entrypoint()
def main(run_id: str='faster-full-001'):
    print(collect.remote(run_id))
