"""Scoped YOLO acquisition and experiment entry points in repro-sign."""
from pathlib import Path
import modal

APP = modal.App('repro-992e7a-yolo')
DATA = modal.Volume.from_name('datasets', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
OUTPUT = modal.Volume.from_name('repro-992e7a-results', create_if_missing=True, version=2)
HERE = Path(__file__).parent
ENV = {'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub'}
CPU_IMAGE = (modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.1.0', 'numpy==1.26.4', 'PyYAML==6.0.2')
             .env(ENV).add_local_file(HERE / 'yolo_data.py', '/work/yolo_data.py')
             .add_local_file(HERE / 'yolo_prepare.py', '/work/yolo_prepare.py')
             .add_local_file(HERE / 'data.sh', '/work/data.sh'))


def data_job(run_id: str, mode: str):
    import datetime
    import hashlib
    import json
    import os
    import subprocess
    import time
    assert mode in {'acquire', 'prepare'}
    script = 'yolo_data.py' if mode == 'acquire' else 'yolo_prepare.py'
    ceiling = 1740 if mode == 'acquire' else 840
    output = Path('/outputs') / run_id
    if (output / 'execution.json').exists():
        existing = (output / 'execution.json').read_text()
        if json.loads(existing)['exit_code'] != 0:
            raise RuntimeError('Retained acquisition failed; use a separately declared attempt ID')
        return existing
    if (output / 'started.json').exists() or (output / 'console.log').exists():
        raise RuntimeError('Interrupted acquisition exists; refusing automatic replay or log overwrite')
    output.mkdir(parents=True, exist_ok=True)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / 'started.json').write_text(json.dumps({
        'started_at_utc': start, 'modal_app_id': APP.app_id,
        'modal_function_call_id': modal.current_function_call_id(),
        'max_wall_seconds': ceiling + 60,
    }, indent=2) + '\n')
    OUTPUT.commit()
    (output / 'executed-source.py').write_bytes(Path('/work/' + script).read_bytes())
    with (output / 'freeze.txt').open('w') as freeze:
        subprocess.run(['python3', '-m', 'pip', 'freeze'], stdout=freeze, check=True)
    clock = time.monotonic()
    code = 1
    try:
        with (output / 'console.log').open('w') as log:
            code = subprocess.run(['python3', '/work/' + script], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=ceiling).returncode
    except subprocess.TimeoutExpired:
        code = 124
        raise
    finally:
        record = {'run_id': run_id, 'started_at_utc': start,
                  'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'wall_seconds': time.monotonic() - clock, 'exit_code': code,
                  'modal_app_id': APP.app_id,
                  'modal_function_call_id': modal.current_function_call_id(),
                  'modal_task_id': os.environ.get('MODAL_TASK_ID'),
                  'source_sha256': hashlib.sha256(Path('/work/' + script).read_bytes()).hexdigest()}
        manifest = Path('/datasets/belmadoui-arabic-sign-language/acquisition-manifest.json')
        if manifest.exists():
            payload = manifest.read_bytes()
            (output / 'acquisition-manifest.json').write_bytes(payload)
            record['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
        final_manifest = Path('/datasets/belmadoui-arabic-sign-language/manifest.json')
        if mode == 'prepare' and final_manifest.exists():
            payload = final_manifest.read_bytes()
            (output / 'manifest.json').write_bytes(payload)
            record['prepared_manifest_sha256'] = hashlib.sha256(payload).hexdigest()
        record['artifact_sha256'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir() if path.is_file() and path.name != 'execution.json'
        }
        (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        DATA.commit()
        OUTPUT.commit()
    if code:
        raise RuntimeError(f'Dataset acquisition exited {code}; see retained console')
    return json.dumps(record)


@APP.function(image=CPU_IMAGE, cpu=2, memory=4096, timeout=1800, retries=0,
              volumes={'/datasets': DATA, '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def acquire(run_id: str):
    return data_job(run_id, 'acquire')


@APP.function(image=CPU_IMAGE, cpu=2, memory=4096, timeout=900, retries=0,
              volumes={'/datasets': DATA, '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def prepare(run_id: str):
    return data_job(run_id, 'prepare')


@APP.local_entrypoint()
def main(run_id: str = 'data-acquisition-v1', mode: str = 'acquire'):
    assert mode in {'acquire', 'prepare'}
    print((acquire if mode == 'acquire' else prepare).remote(run_id))
