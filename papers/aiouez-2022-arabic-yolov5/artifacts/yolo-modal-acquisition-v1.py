"""Scoped YOLO acquisition and experiment entry points in repro-sign."""
from pathlib import Path
import modal

APP = modal.App('repro-992e7a-yolo')
DATA = modal.Volume.from_name('datasets', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
OUTPUT = modal.Volume.from_name('repro-992e7a-results', create_if_missing=True, version=2)
HERE = Path(__file__).parent
ENV = {'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub'}
CPU_IMAGE = (modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.1.0')
             .env(ENV).add_local_file(HERE / 'yolo_data.py', '/work/yolo_data.py')
             .add_local_file(HERE / 'data.sh', '/work/data.sh'))


@APP.function(image=CPU_IMAGE, cpu=2, memory=4096, timeout=1800, retries=0,
              volumes={'/datasets': DATA, '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def acquire(run_id: str):
    import datetime
    import hashlib
    import json
    import os
    import subprocess
    import time
    output = Path('/outputs') / run_id
    if (output / 'execution.json').exists():
        return (output / 'execution.json').read_text()
    output.mkdir(parents=True, exist_ok=True)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.monotonic()
    code = 1
    try:
        with (output / 'console.log').open('w') as log:
            code = subprocess.run(['bash', '/work/data.sh'], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=1740).returncode
    finally:
        record = {'run_id': run_id, 'started_at_utc': start,
                  'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'wall_seconds': time.monotonic() - clock, 'exit_code': code,
                  'modal_app_id': os.environ.get('MODAL_APP_ID'),
                  'modal_function_call_id': os.environ.get('MODAL_FUNCTION_CALL_ID'),
                  'modal_task_id': os.environ.get('MODAL_TASK_ID'),
                  'source_sha256': hashlib.sha256(Path('/work/yolo_data.py').read_bytes()).hexdigest()}
        manifest = Path('/datasets/belmadoui-arabic-sign-language/acquisition-manifest.json')
        if manifest.exists():
            payload = manifest.read_bytes()
            (output / 'acquisition-manifest.json').write_bytes(payload)
            record['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
        (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        DATA.commit()
        OUTPUT.commit()
    if code:
        raise RuntimeError(f'Dataset acquisition exited {code}; see retained console')
    return json.dumps(record)


@APP.local_entrypoint()
def main(run_id: str = 'data-acquisition-v1'):
    print(acquire.remote(run_id))
