"""One bounded native YOLO full run with immutable claim and checkpoint recovery."""
from pathlib import Path
import modal
APP = modal.App('repro-992e7a-yolo-full')
HERE = Path(__file__).parent
DATA = modal.Volume.from_name('datasets', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
OUTPUT = modal.Volume.from_name('repro-992e7a-results', version=2)
MANIFEST_SHA = 'bdb2b2a87d2b93af07d977dafac14cf5f99c2e3333e97350cada17da8475356e'
IMAGE = (modal.Image.from_dockerfile(HERE.parent / 'Dockerfile.yolo')
         .env({'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
         .add_local_file(HERE / 'yolo_stage.py', '/work/yolo_stage.py')
         .add_local_file(HERE / 'yolo_train.py', '/work/yolo_train.py')
         .add_local_file(Path(__file__), '/work/yolo_full.py')
         .add_local_file(HERE.parent / 'Dockerfile.yolo', '/work/Dockerfile.yolo')
         .add_local_dir(HERE.parent / 'patches', '/work/patches'))


@APP.function(image=IMAGE, gpu='A100-80GB', cpu=4, memory=65536, timeout=14400,
              retries=0, volumes={'/datasets': DATA.read_only(), '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def full(run_id: str, model: str, weights_sha256: str, max_wall_seconds: int):
    import datetime
    import hashlib
    import json
    import os
    import re
    import shutil
    import signal
    import subprocess
    import threading
    import time
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', run_id), 'Invalid run ID'
    assert re.fullmatch(r'[a-f0-9]{64}', weights_sha256), 'Invalid weight hash'
    assert model in ['s', 'm', 'l'] and 600 <= max_wall_seconds <= 14400
    output = Path('/outputs') / run_id
    output.mkdir(exist_ok=True)
    def stamp():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    source_paths = [Path('/work') / name for name in ['yolo_stage.py', 'yolo_train.py', 'yolo_full.py', 'Dockerfile.yolo']] + sorted(Path('/work/patches').glob('yolo-*.patch'))
    identity = {'call_id': modal.current_function_call_id(), 'model': model,
                'weights_sha256': weights_sha256, 'manifest_sha256': MANIFEST_SHA,
                'max_wall_seconds': max_wall_seconds,
                'sources': {str(p.relative_to('/work')): digest(p) for p in source_paths}}
    claim_path = output / 'claim.json'
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        claim = json.loads(claim_path.read_text())
        assert claim['identity'] == identity, 'Refuse duplicate call/source change/deadline reset.'
    else:
        clock = time.time()
        claim = {'identity': identity, 'started_at_utc': stamp(), 'started_unix': clock,
                 'deadline_unix': clock + max_wall_seconds, 'max_segments': 4, 'app_id': APP.app_id}
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(claim, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        OUTPUT.commit()
    execution_path = output / 'execution.json'
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {'segments': []}
    if execution.get('exit_code') == 0:
        return execution
    def save():
        temporary = execution_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(execution, indent=2) + '\n')
        temporary.replace(execution_path)
        OUTPUT.commit()
    remaining = claim['deadline_unix'] - time.time()
    if remaining <= 120 or len(execution['segments']) >= claim['max_segments']:
        execution.update(exit_code=124, finished_at_utc=stamp(), stop_detail='Original deadline or segment ceiling exhausted.')
        save()
        return execution
    if execution['segments']:
        if execution['segments'][-1].get('exit_code') is None:
            execution['segments'][-1]['state'] = 'interrupted'
        assert (output / model / 'recovery.json').exists(), 'No verified checkpoint; fresh restart forbidden.'
    segment = {'number': len(execution['segments']) + 1, 'started_at_utc': stamp(),
               'state': 'running', 'exit_code': None, 'modal_task_id': os.environ.get('MODAL_TASK_ID')}
    execution['segments'].append(segment)
    execution.update(run_id=run_id, model=model, started_at_utc=claim['started_at_utc'],
                     deadline_unix=claim['deadline_unix'], modal_app_id=APP.app_id,
                     modal_function_call_id=modal.current_function_call_id(), exit_code=None,
                     gpu='A100-80GB', gpu_count=1, cpu=4, memory_mib=65536)
    save()
    child = None
    def terminate():
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    watchdog = threading.Timer(max(0, claim['deadline_unix'] - time.time()), lambda: (terminate(), os._exit(124)))
    watchdog.daemon = True
    watchdog.start()
    code = None
    try:
        sources = output / 'sources'
        if not sources.exists():
            shutil.copytree('/work', sources)
        for patch in sorted(Path('/work/patches').glob('yolo-*.patch')):
            subprocess.run(['git', '-C', '/opt/yolov5', 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', '/opt/yolov5', 'apply', str(patch)], check=True)
        (output / f'freeze-{segment["number"]}.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
        (output / f'gpu-{segment["number"]}.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
        commands = [['python', '/work/yolo_stage.py', '--manifest-sha256', MANIFEST_SHA],
                    ['python', '/work/yolo_train.py', '--model', model, '--data', '/tmp/yolo-data/data.yaml',
                     '--output', str(output / model), '--weights-sha256', weights_sha256]]
        if segment['number'] > 1:
            commands[-1].append('--resume')
        segment['commands'] = commands
        save()
        with (output / f'console-segment-{segment["number"]:02d}.log').open('w') as log:
            for command in commands:
                log.write(json.dumps({'command': command}) + '\n')
                log.flush()
                seconds = int(claim['deadline_unix'] - time.time() - 120)
                if seconds <= 0:
                    code = 124
                    break
                code = None
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    terminate()
                    child.wait()
                    code = 124
                if code:
                    break
    except BaseException:
        # Cancellation is not a fabricated native exit1 and must not orphan the trainer.
        terminate()
        segment['state'] = 'interrupted' if code is None else 'failed'
        raise
    finally:
        terminate()
        segment.update(exit_code=code, finished_at_utc=stamp())
        if code is not None:
            segment['state'] = 'succeeded' if code == 0 else 'stopped' if code == 124 else 'failed'
        execution.update(exit_code=code, finished_at_utc=segment['finished_at_utc'],
                         conservative_elapsed_seconds=time.time() - claim['started_unix'])
        try:
            save()
        finally:
            watchdog.cancel()
    return execution


@APP.local_entrypoint()
def main(run_id: str, model: str, weights_sha256: str, max_wall_seconds: int):
    result = full.remote(run_id, model, weights_sha256, max_wall_seconds)
    print(result)
    if result['exit_code'] != 0:
        raise SystemExit(result['exit_code'] or 1)
