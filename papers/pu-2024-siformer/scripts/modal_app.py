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
         .add_local_file(Path(__file__).with_name('preflight.py'), '/opt/preflight.py'))
datasets = modal.Volume.from_name('datasets')
cache = modal.Volume.from_name('huggingface-cache')
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

@app.local_entrypoint()
def main(run_id: str = 'preflight-001', acquire: bool=False):
    print(acquire_data.remote() if acquire else preflight.remote(run_id))
