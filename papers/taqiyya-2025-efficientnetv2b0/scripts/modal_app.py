"""Bounded container entrypoints; invoke only through modal_repro_sign.sh."""
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent
app=modal.App('repro-c56bc98c-efficientnetv2b0')
datasets=modal.Volume.from_name('datasets',create_if_missing=False,version=2)
cache=modal.Volume.from_name('huggingface-cache',create_if_missing=False,version=2)
results=modal.Volume.from_name('taqiyya-2025-efficientnetv2b0-results',create_if_missing=True,version=2)
ENV={'HF_HOME':'/cache/huggingface','HF_HUB_CACHE':'/cache/huggingface/hub','KERAS_HOME':'/cache/huggingface/keras','TF_CPP_MIN_LOG_LEVEL':'2'}
data_image=modal.Image.debian_slim(python_version='3.11').pip_install('numpy==1.26.4').add_local_file(ROOT/'data.sh','/app/data.sh')
@app.function(image=data_image,cpu=4,memory=8192,timeout=1800,volumes={'/datasets':datasets,'/cache/huggingface':cache},env=ENV)
def populate():
    import subprocess
    subprocess.run(['bash','/app/data.sh'],check=True)
    datasets.commit()
    return Path('/datasets/mavi-27-class/manifest.json').read_text()
train_image=modal.Image.from_dockerfile(ROOT.parent/'Dockerfile').add_local_file(ROOT/'train.py','/app/train.py')
@app.function(image=train_image,gpu='A10G',cpu=8,memory=32768,timeout=28800,volumes={'/datasets':datasets.with_mount_options(read_only=True),'/cache/huggingface':cache,'/results':results},env=ENV)
def train(model: str, preflight: bool, gpu_label: str='A10G'):
    import glob, json, os, subprocess
    run_id=('preflight-' if preflight else 'full-')+model+'-seed42-v3'+('-a100' if gpu_label=='A100' else '')
    output=Path('/results')/run_id
    command=['/opt/taqiyya/bin/python','/app/train.py','--model',model,'--out',str(output)]
    if preflight:command.append('--preflight')
    os.environ['LD_LIBRARY_PATH']=':'.join(glob.glob('/opt/taqiyya/lib/python*/site-packages/nvidia/*/lib'))+':'+os.environ.get('LD_LIBRARY_PATH','')
    try:
        subprocess.run(command,check=True,timeout=(550 if gpu_label=='A100' else 3500) if preflight else (28700 if model=='convnexttiny' else 21500))
    finally:
        results.commit()
    return (output/'metrics.json').read_text()
@app.local_entrypoint()
def main(mode: str='data', model: str='efficientnetv2b0', gpu: str='A10G'):
    assert gpu in ('A10G','A100')
    if mode=='data':print(populate.remote())
    else:
        ceiling=(600 if gpu=='A100' else 3600) if mode=='preflight' else (28800 if model=='convnexttiny' else 21600)
        call=train.with_options(gpu=gpu,timeout=ceiling).spawn(model,mode=='preflight',gpu)
        print('Function call:',call.object_id,flush=True)
        print(call.get())
