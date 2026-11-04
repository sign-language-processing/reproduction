"""Bounded Modal jobs for the approved conditional reconstruction."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('arsl-4982792c-conditional')
image=(modal.Image.from_registry('nvidia/cuda@sha256:081a31f56decc6c460df1808ed1c4867fb30b0fbfa8929258b10e3e5d6dc1a2e',add_python='3.10')
    .pip_install('tensorflow==2.8.0','keras==2.8.0','numpy==1.23.5','protobuf==3.20.3',
        'tensorflow-addons==0.16.1','tensorflow-hub==0.12.0','tensorflow-datasets==4.8.3',
        'tensorflow-metadata==1.12.0','keras-cv-attention-models==1.3.17','vit-keras==0.1.2',
        'scipy==1.10.1','scikit-learn==1.3.2','pyarrow==14.0.2','pillow==10.4.0','h5py==3.10.0')
    .pip_install('opencv-python-headless==4.8.1.78')
    .env({'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub',
        'KERAS_HOME':'/cache/huggingface/keras-arsl-tf28','TFHUB_CACHE_DIR':'/cache/huggingface/tfhub',
        'PYTHONUNBUFFERED':'1','TF_CPP_MIN_LOG_LEVEL':'1','TF_FORCE_GPU_ALLOW_GROWTH':'true','PYTHONHASHSEED':'42',
        'TF_ENABLE_ONEDNN_OPTS':'0','OMP_NUM_THREADS':'8'})
    .add_local_dir(ROOT,'/app'))
datasets=modal.Volume.from_name('datasets',version=2)
cache=modal.Volume.from_name('huggingface-cache',version=2)
results=modal.Volume.from_name('arsl-4982792c-results',create_if_missing=True,version=2)
@app.function(image=image,cpu=2,memory=8192,timeout=600,
    volumes={'/datasets':datasets,'/cache/huggingface':cache,'/results':results})
def prepare():
    import subprocess,json
    p=subprocess.run(['python','/app/data.py'],capture_output=True,text=True)
    print(p.stdout,p.stderr)
    if p.returncode:raise RuntimeError('data preparation failed')
    datasets.commit()
    out={'data':json.loads(p.stdout.strip().splitlines()[-1]),'image_id':image.object_id,'function_call_id':modal.current_function_call_id()}
    Path('/results/preparation.json').write_text(json.dumps(out,indent=2))
    results.commit()
    return out
@app.function(image=image,gpu='A100-40GB',cpu=8,memory=32768,timeout=86400,retries=0,max_containers=1,
    volumes={'/datasets':datasets.read_only(),'/cache/huggingface':cache,'/results':results})
def execute(model:str,mode:str,run_id:str,resume_from:str='',max_seconds:int=900,preflight_id:str=''):
    import subprocess,json,time,threading
    output=Path('/results')/run_id
    output.mkdir(parents=True,exist_ok=True)
    metadata={'image_id':image.object_id,'function_call_id':modal.current_function_call_id(),'model':model,'mode':mode,'run_id':run_id}
    def save_metadata():
        temporary=output/'modal.json.tmp'
        temporary.write_text(json.dumps(metadata,indent=2));temporary.replace(output/'modal.json')
    save_metadata();results.commit()
    command=['python','/app/train.py','--model',model,'--mode',mode,'--output',str(output)]
    if resume_from:command+=['--resume-from','/results/'+resume_from]
    if preflight_id:command+=['--pins-from','/results/'+preflight_id]
    start=time.monotonic()
    with (output/'console.log').open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert 0<max_seconds<=82800
        timer=threading.Timer(max_seconds,process.kill)
        timer.start()
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        rc=process.wait();timer.cancel()
    metadata.update(exit_code=rc,wall_seconds=time.monotonic()-start)
    save_metadata();results.commit();cache.commit()
    if rc:raise RuntimeError(f'{model} {mode} exited {rc}; retained logs at {output}')
    result=json.loads((output/'metrics.json').read_text());result['modal']=metadata
    return result
@app.function(image=image,cpu=1,memory=4096,timeout=900,
    volumes={'/results':results.read_only(),'/cache/huggingface':cache})
def inspect_outputs(interrupted_run_ids:list[str]):
    import json,hashlib
    evidence={}
    for directory in sorted(Path('/results').iterdir()):
        if not directory.is_dir() or not (directory/'modal.json').exists():continue
        try:state=json.loads((directory/'modal.json').read_text())
        except json.JSONDecodeError:
            print('Skipping concurrently written metadata:',directory.name)
            continue
        interrupted='exit_code' not in state
        if interrupted and directory.name not in interrupted_run_ids:continue
        files={}
        for name in ['metrics.json','configuration.json','weights.json','modal.json','environment.txt','artifact-hashes.json']:
            p=directory/name
            if p.exists():
                data=p.read_bytes()
                files[name]={'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data),'content':json.loads(data) if name.endswith('.json') else data.decode()}
        if (directory/'console.log').exists():
            raw=(directory/'console.log').read_bytes()
            files['console.log']={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'content':raw.decode(errors='replace')[-16000:] if state.get('exit_code',1) else ''}
        if interrupted:
            import tensorflow as tf,numpy as np
            import sys
            sys.path.insert(0,'/app')
            from data import sha
            latest=tf.train.latest_checkpoint(str(directory/'checkpoints'))
            inspection={'latest_checkpoint':latest,'all_checkpoint_tensors_readable':False}
            if latest:
                reader=tf.train.load_checkpoint(latest);variables=tf.train.list_variables(latest)
                scalars={}
                for name,shape in variables:
                    value=reader.get_tensor(name)
                    if hasattr(value,'dtype') and np.issubdtype(value.dtype,np.floating):assert np.isfinite(value).all(),name
                    if name in ['epoch/.ATTRIBUTES/VARIABLE_VALUE','batch/.ATTRIBUTES/VARIABLE_VALUE','optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE']:scalars[name]=int(value)
                    del value
                inspection.update(all_checkpoint_tensors_readable=True,tensor_count=len(variables),scalars=scalars)
                del reader
            inspection['native_file_sha256']={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
            files['_checkpoint_inspection']=inspection
        evidence[directory.name]=files
    return evidence

@app.local_entrypoint()
def main(model:str='mobilenet',mode:str='prepare',run_id:str='prepare',resume_from:str='',max_seconds:int=900,preflight_id:str='',include_interrupted:bool=False):
    if mode=='report':
        import json
        ledger=json.loads((ROOT.parent/'reproduction.json').read_text())
        interrupted_ids=[r['run_id'] for r in ledger['runs'] if r.get('terminal',{}).get('state')=='interrupted'] if include_interrupted else []
        evidence=inspect_outputs.remote(interrupted_ids)
        Path('/tmp/arsl-evidence.json').write_text(json.dumps(evidence,indent=2))
        print('Collected',len(evidence),'output directories into /tmp/arsl-evidence.json')
    elif mode=='prepare':print(prepare.remote())
    else:print(execute.remote(model,mode,run_id,resume_from,max_seconds,preflight_id))
