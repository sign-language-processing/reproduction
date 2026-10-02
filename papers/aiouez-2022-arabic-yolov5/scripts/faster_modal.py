"""Bounded Detectron2 source probe/preflight; invoke only with repro-sign wrapper."""
from pathlib import Path
import modal

APP = modal.App('repro-992e7a-faster')
DATA = modal.Volume.from_name('datasets', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
OUTPUT = modal.Volume.from_name('repro-992e7a-results', create_if_missing=True, version=2)
HERE = Path(__file__).parent
BASE = 'ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291'
ENV = {'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub',
       'MAX_JOBS': '2', 'FORCE_CUDA': '0'}
IMAGE = (modal.Image.from_registry(BASE).apt_install('git')
         .pip_install('fvcore==0.1.5.post20221221', 'iopath==0.1.9', 'yacs==0.1.8',
                      'pycocotools==2.0.11', 'omegaconf==2.3.0', 'hydra-core==1.3.2',
                      'termcolor==2.5.0', 'tensorboard==2.19.0', 'cloudpickle==3.1.1')
         .env(ENV)
         .add_local_file(HERE / 'faster_run.py', '/work/faster_run.py')
         .add_local_file(HERE.parent / 'faster-pillow.patch', '/work/faster-pillow.patch')
         .add_local_file(HERE / 'faster_preflight.py', '/work/faster_preflight.py'))


@APP.function(image=IMAGE, cpu=4, memory=16384, timeout=300, retries=0,
              volumes={'/datasets': DATA.with_mount_options(read_only=True),
                       '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def source_probe(run_id: str, wheel_run: str):
    import datetime
    import hashlib
    import json
    import os
    import subprocess
    import time
    output = Path('/outputs') / run_id
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.monotonic()
    wheel = Path('/outputs') / wheel_run / 'wheels/detectron2-0.6-cp312-cp312-linux_x86_64.whl'
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '8c66da87f7a725c03990caa9dde11ce69610a63f8808b7b27e782d3fbcc2b328'
    command = ['bash', '-c', 'set -eu\n'
               'python -m pip install --no-deps "$2"\n'
               'git apply --unsafe-paths --directory="$(python -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')" /work/faster-pillow.patch\n'
               'python /work/faster_run.py --probe --output "$1"', 'faster-source-probe', str(output), str(wheel)]
    record = {'run_id': run_id, 'started_at_utc': start,
              'modal_app_id': APP.app_id, 'modal_function_call_id': modal.current_function_call_id(),
              'modal_task_id': os.environ.get('MODAL_TASK_ID'), 'command': command,
              'gpu_count': 0, 'base_image': BASE, 'native_timeout_seconds': 240, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
    (output / 'faster_run.py').write_bytes(Path('/work/faster_run.py').read_bytes())
    (output / 'faster-pillow.patch').write_bytes(Path('/work/faster-pillow.patch').read_bytes())
    (output / 'execution.json').write_text(json.dumps(record, indent=2))
    OUTPUT.commit()
    code = 1
    try:
        with (output / 'console.log').open('w') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
            except subprocess.TimeoutExpired:
                code = 124
    finally:
        (output / 'pip-freeze.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
        record.update(exit_code=code, wall_seconds=time.monotonic()-clock,
                      finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (output / 'execution.json').write_text(json.dumps(record, indent=2)+'\n')
        files = []
        for path in sorted(output.rglob('*')):
            if path.is_file():
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
                files.append({'path': str(path.relative_to(output)), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
        (output / 'evidence.json').write_text(json.dumps(files, indent=2)+'\n')
        OUTPUT.commit()
    print((output / 'console.log').read_text()[-14000:])
    return record


@APP.function(image=IMAGE, gpu='A100-80GB', cpu=4, memory=65536, timeout=1800, retries=0,
              volumes={'/datasets': DATA.with_mount_options(read_only=True),
                       '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def preflight(run_id: str, data_root: str, train_json: str, val_json: str,
              manifest: str, manifest_sha256: str):
    import datetime
    import hashlib
    import json
    import os
    import subprocess
    import time
    output = Path('/outputs') / run_id
    output.mkdir(parents=True, exist_ok=False)  # Automatic replay must not restart training.
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.monotonic()
    wheel = Path('/outputs/faster-source-probe-001/wheels/detectron2-0.6-cp312-cp312-linux_x86_64.whl')
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '8c66da87f7a725c03990caa9dde11ce69610a63f8808b7b27e782d3fbcc2b328'
    native = ['python', '/work/faster_preflight.py', '--output', str(output),
              '--data-root', data_root, '--train-json', train_json, '--val-json', val_json,
              '--manifest', manifest, '--manifest-sha256', manifest_sha256]
    setup = ['bash', '-c', 'set -eu\n'
             'python -m pip install --no-deps "$1"\n'
             'git apply --unsafe-paths --directory="$(python -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')" /work/faster-pillow.patch\n'
             'git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2 /opt/detectron2\n'
             'test "$(git -C /opt/detectron2 rev-parse HEAD)" = d1e04565d3bec8719335b88be9e9b961bf3ec464\n'
             'shift\nexec "$@"', 'faster-preflight', str(wheel)] + native
    record = {'run_id': run_id, 'started_at_utc': start, 'modal_app_id': APP.app_id,
              'modal_function_call_id': modal.current_function_call_id(),
              'modal_task_id': os.environ.get('MODAL_TASK_ID'), 'command': setup,
              'gpu_count': 1, 'gpu': 'A100-80GB', 'base_image': BASE,
              'native_timeout_seconds': 1740, 'dataset_manifest_sha256': manifest_sha256}
    for name in ('faster_preflight.py', 'faster-pillow.patch'):
        (output/name).write_bytes(Path('/work',name).read_bytes())
    (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    (output/'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    OUTPUT.commit()
    code = 1
    try:
        with (output/'console.log').open('w') as log:
            try:
                code = subprocess.run(setup,stdout=log,stderr=subprocess.STDOUT,timeout=1740).returncode
            except subprocess.TimeoutExpired:
                code = 124
    finally:
        (output/'pip-freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
        record.update(exit_code=code,wall_seconds=time.monotonic()-clock,
                      finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
        files=[]
        for path in sorted(output.rglob('*')):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
                files.append({'path':str(path.relative_to(output)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
        (output/'evidence.json').write_text(json.dumps(files,indent=2)+'\n')
        OUTPUT.commit()
    print((output/'console.log').read_text()[-24000:])
    return record


@APP.local_entrypoint()
def main(run_id: str = 'faster-source-probe-003', wheel_run: str = 'faster-source-probe-001',
         gpu_preflight: bool = False, data_root: str = '', train_json: str = '',
         val_json: str = '', manifest: str = '', manifest_sha256: str = ''):
    if gpu_preflight:
        assert all((data_root, train_json, val_json, manifest, manifest_sha256))
        result = preflight.remote(run_id, data_root, train_json, val_json, manifest, manifest_sha256)
    else:
        result = source_probe.remote(run_id, wheel_run)
    print(result)
    if result['exit_code']:
        raise SystemExit(result['exit_code'])
