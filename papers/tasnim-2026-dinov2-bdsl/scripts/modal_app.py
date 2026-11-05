"""Invoke through modal_repro_sign.sh; all storage scoped to repro-sign."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('repro-8755046-dinov2-bdsl')
datasets=modal.Volume.from_name('datasets',create_if_missing=False,version=2)
cache=modal.Volume.from_name('huggingface-cache',create_if_missing=False,version=2)
ENV={'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub','TORCH_HOME':'/cache/huggingface/torch'}
data_image=modal.Image.debian_slim(python_version='3.12').add_local_file(ROOT/'data.sh','/app/data.sh')
@app.function(image=data_image,cpu=4,memory=8192,timeout=1800,volumes={'/datasets':datasets,'/cache/huggingface':cache},env=ENV)
def populate():
    import subprocess
    try:subprocess.run(['bash','/app/data.sh'],check=True)
    finally:datasets.commit()
    return Path('/datasets/bdsl49-v6-recognition/manifest.json').read_text()
@app.function(image=data_image,cpu=1,memory=1024,timeout=300,volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache},env=ENV)
def verify_upload(filename: str):
    import hashlib
    path=Path('/datasets/bdsl49-v6-recognition')/filename
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return {'filename':filename,'bytes':path.stat().st_size,'sha256':h.hexdigest()}
results=modal.Volume.from_name('tasnim-2026-dinov2-bdsl-results',create_if_missing=True,version=2)
train_image=modal.Image.from_dockerfile(ROOT.parent/'Dockerfile').add_local_file(ROOT/'train.py','/app/train.py')
@app.function(image=train_image,gpu='A10G',cpu=4,memory=16384,timeout=14400,volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/results':results},env=ENV)
def train(preflight: bool):
    import subprocess
    output=Path('/results')/('preflight-seed42' if preflight else 'full-seed42')
    command=['python','/app/train.py','--out',str(output)]
    if preflight:command.append('--preflight')
    try:subprocess.run(command,check=True)
    finally:results.commit()
    return (output/'metrics.json').read_text()
@app.local_entrypoint()
def main(mode: str='data', filename: str='Recognition_2.zip'):
    if mode=='verify-upload':call=verify_upload.spawn(filename)
    elif mode=='data':call=populate.spawn()
    else:call=train.spawn(mode=='preflight')
    print('Function call:',call.object_id,flush=True);print(call.get())
