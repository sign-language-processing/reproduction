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
                      'termcolor==2.5.0', 'tensorboard==2.19.0')
         .env(ENV)
         .add_local_file(HERE / 'faster_run.py', '/work/faster_run.py'))


@APP.function(image=IMAGE, cpu=4, memory=16384, timeout=3600, retries=0,
              volumes={'/datasets': DATA.with_mount_options(read_only=True),
                       '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def source_probe(run_id: str):
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
    command = ['bash', '-c', 'set -eu\n'
               'git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2 /opt/detectron2\n'
               'test "$(git -C /opt/detectron2 rev-parse HEAD)" = d1e04565d3bec8719335b88be9e9b961bf3ec464\n'
               'python -m pip wheel --no-build-isolation --no-deps /opt/detectron2 --wheel-dir "$1/wheels"\n'
               'python -m pip install --no-deps "$1"/wheels/*.whl\n'
               'python /work/faster_run.py --probe --output "$1"', 'faster-source-probe', str(output)]
    record = {'run_id': run_id, 'started_at_utc': start,
              'modal_app_id': APP.app_id, 'modal_function_call_id': modal.current_function_call_id(),
              'modal_task_id': os.environ.get('MODAL_TASK_ID'), 'command': command,
              'gpu_count': 0, 'base_image': BASE, 'native_timeout_seconds': 3480}
    (output / 'execution.json').write_text(json.dumps(record, indent=2))
    OUTPUT.commit()
    code = 1
    try:
        with (output / 'console.log').open('w') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3480).returncode
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


@APP.local_entrypoint()
def main(run_id: str = 'faster-source-probe-001'):
    result = source_probe.remote(run_id)
    print(result)
    if result['exit_code']:
        raise SystemExit(result['exit_code'])
