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
       'MAX_JOBS': '2', 'FORCE_CUDA': '0', 'OMP_NUM_THREADS': '4',
       'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
IMAGE = (modal.Image.from_registry(BASE).apt_install('git')
         .pip_install('fvcore==0.1.5.post20221221', 'iopath==0.1.9', 'yacs==0.1.8',
                      'pycocotools==2.0.11', 'omegaconf==2.3.0', 'hydra-core==1.3.2',
                      'termcolor==2.5.0', 'tensorboard==2.19.0', 'cloudpickle==3.1.1')
         .env(ENV)
         .add_local_file(HERE / 'faster_run.py', '/work/faster_run.py')
         .add_local_file(HERE.parent / 'faster-pillow.patch', '/work/faster-pillow.patch')
         .add_local_file(HERE / 'faster_preflight.py', '/work/faster_preflight.py')
         .add_local_file(HERE / 'faster_train.py', '/work/faster_train.py')
         .add_local_file(HERE / 'faster_modal.py', '/work/faster_modal.py'))


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


@APP.function(image=IMAGE, gpu='A100-80GB', cpu=4, memory=65536, timeout=86400, retries=0,
              volumes={'/datasets': DATA.with_mount_options(read_only=True),
                       '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def full(run_id: str, data_root: str, manifest_sha256: str, weights_file: str,
         weights_sha256: str, max_wall_seconds: int):
    import datetime
    import hashlib
    import json
    import os
    import signal
    import subprocess
    import threading
    import time
    assert 600 <= max_wall_seconds <= 86400
    output = Path('/outputs')/run_id
    output.mkdir(parents=True,exist_ok=True)
    call_id = modal.current_function_call_id()
    now = time.time()
    identity = {'function_call_id':call_id,'manifest_sha256':manifest_sha256,
                'weights_sha256':weights_sha256,'weights_file':weights_file,'data_root':data_root,
                'max_wall_seconds':max_wall_seconds,
                'runner_sha256':hashlib.sha256(Path('/work/faster_train.py').read_bytes()).hexdigest(),
                'launcher_sha256':hashlib.sha256(Path('/work/faster_modal.py').read_bytes()).hexdigest(),
                'patch_sha256':hashlib.sha256(Path('/work/faster-pillow.patch').read_bytes()).hexdigest()}
    claim_path = output/'claim.json'
    try:
        descriptor = os.open(claim_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        claim = json.loads(claim_path.read_text())
        assert claim['identity'] == identity, 'Existing logical run cannot be reset or concurrently relaunched.'
    else:
        claim = {'identity':identity,'started_unix':now,'deadline_unix':now+max_wall_seconds,
                 'started_at_utc':datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
                 'modal_app_id':APP.app_id,'max_segments':4}
        with os.fdopen(descriptor,'w') as stream:
            json.dump(claim,stream,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        OUTPUT.commit()
    execution_path=output/'execution.json'
    execution=json.loads(execution_path.read_text()) if execution_path.exists() else {'segments':[]}
    def save():
        temporary=execution_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(execution,indent=2)+'\n')
        temporary.replace(execution_path)
        OUTPUT.commit()
    if execution.get('exit_code') == 0:
        return execution
    if execution['segments'] and execution['segments'][-1].get('exit_code') is None:
        execution['segments'][-1].update(state='interrupted',
            interruption_observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            detail='Prior container ended without a terminal record; exact native exit/time unknown.')
    remaining = claim['deadline_unix']-time.time()
    if remaining <= 0 or len(execution['segments']) >= claim['max_segments']:
        execution.update(exit_code=124,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         stop_detail='Original wall ceiling or initial segment plus three provider recoveries exhausted.',
                         conservative_elapsed_seconds=time.time()-claim['started_unix'])
        save()
        return execution
    # This watchdog only terminates its own container process, never a replacement.
    watchdog = threading.Timer(remaining,lambda:os._exit(124))
    watchdog.daemon=True
    watchdog.start()
    try:
        segment={'number':len(execution['segments'])+1,'state':'running','exit_code':None,
                 'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'modal_task_id':os.environ.get('MODAL_TASK_ID')}
        execution['segments'].append(segment)
        execution.update(run_id=run_id,modal_app_id=APP.app_id,modal_function_call_id=call_id,
                         started_at_utc=claim['started_at_utc'],deadline_unix=claim['deadline_unix'],
                         max_wall_seconds=max_wall_seconds,gpu='A100-80GB',gpu_count=1,cpu=4,memory_mib=65536,
                         base_image=BASE,source_revision='d1e04565d3bec8719335b88be9e9b961bf3ec464',
                         accounting='Original elapsed wall-clock includes interrupted/idle gaps; conservative allocation bound, not exact billing.')
        save()
        for name in ('faster_train.py','faster_modal.py','faster-pillow.patch'):
            destination=output/name
            payload=Path('/work',name).read_bytes()
            if destination.exists():assert destination.read_bytes() == payload
            else:destination.write_bytes(payload)
        wheel=Path('/outputs/faster-source-probe-001/wheels/detectron2-0.6-cp312-cp312-linux_x86_64.whl')
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '8c66da87f7a725c03990caa9dde11ce69610a63f8808b7b27e782d3fbcc2b328'
        command=['bash','-c','set -eu\n'
                 'python -m pip install --no-deps "$1"\n'
                 'git apply --unsafe-paths --directory="$(python -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')" /work/faster-pillow.patch\n'
                 'git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2 /opt/detectron2\n'
                 'test "$(git -C /opt/detectron2 rev-parse HEAD)" = d1e04565d3bec8719335b88be9e9b961bf3ec464\n'
                 'shift\nexec "$@"','faster-full',str(wheel),'python','/work/faster_train.py',
                 '--data-root',data_root,'--manifest-sha256',manifest_sha256,'--weights-file',weights_file,
                 '--weights-sha256',weights_sha256,'--output',str(output)]
        segment['command']=command
        segment['remaining_native_timeout_seconds']=max(0,int(claim['deadline_unix']-time.time()-120))
        save()
        log_path=output/('console-segment-%02d.log'%segment['number'])
        (output/('gpu-segment-%02d.txt'%segment['number'])).write_text(subprocess.check_output(['nvidia-smi'],text=True))
        code=1
        child=None
        try:
            if segment['remaining_native_timeout_seconds'] <= 0:
                code=124
                raise TimeoutError('Original deadline has no execution time remaining.')
            with log_path.open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=segment['remaining_native_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                    code=124
        finally:
            segment.update(state='succeeded' if code==0 else 'stopped' if code==124 else 'failed',exit_code=code,
                           finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            execution.update(exit_code=code,finished_at_utc=segment['finished_at_utc'],
                             conservative_elapsed_seconds=time.time()-claim['started_unix'])
            (output/('pip-freeze-segment-%02d.txt'%segment['number'])).write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
            save()
        return execution
    finally:
        watchdog.cancel()


@APP.local_entrypoint()
def main(run_id: str = 'faster-source-probe-003', wheel_run: str = 'faster-source-probe-001',
         gpu_preflight: bool = False, data_root: str = '', train_json: str = '',
         val_json: str = '', manifest: str = '', manifest_sha256: str = '',
         full_run: bool = False, weights_file: str = '', weights_sha256: str = '', max_wall_seconds: int = 0):
    if full_run:
        assert not gpu_preflight
        assert all((data_root,manifest_sha256,weights_file,weights_sha256,max_wall_seconds))
        result = full.remote(run_id,data_root,manifest_sha256,weights_file,weights_sha256,max_wall_seconds)
    elif gpu_preflight:
        assert all((data_root, train_json, val_json, manifest, manifest_sha256))
        result = preflight.remote(run_id, data_root, train_json, val_json, manifest, manifest_sha256)
    else:
        result = source_probe.remote(run_id, wheel_run)
    print(result)
    if result['exit_code']:
        raise SystemExit(result['exit_code'])
