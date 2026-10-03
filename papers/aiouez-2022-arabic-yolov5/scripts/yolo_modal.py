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
             .add_local_file(HERE / 'yolo_label_audit.py', '/work/yolo_label_audit.py')
             .add_local_file(HERE / 'data.sh', '/work/data.sh'))


def data_job(run_id: str, mode: str):
    import datetime
    import hashlib
    import json
    import os
    import subprocess
    import time
    assert mode in {'acquire', 'prepare', 'audit'}
    script, ceiling = {'acquire': ('yolo_data.py', 1740), 'prepare': ('yolo_prepare.py', 840),
                       'audit': ('yolo_label_audit.py', 240)}[mode]
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
    clock = time.monotonic()
    (output / 'started.json').write_text(json.dumps({
        'started_at_utc': start, 'modal_app_id': APP.app_id,
        'modal_function_call_id': modal.current_function_call_id(),
        'max_wall_seconds': ceiling + 60,
    }, indent=2) + '\n')
    OUTPUT.commit()
    (output / 'executed-source.py').write_bytes(Path('/work/' + script).read_bytes())
    with (output / 'freeze.txt').open('w') as freeze:
        subprocess.run(['python3', '-m', 'pip', 'freeze'], stdout=freeze, check=True)
    code = 1
    try:
        with (output / 'console.log').open('w') as log:
            command = ['python3', '/work/' + script]
            if mode == 'audit':
                command += ['--output', str(output / 'label-audit.json')]
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
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
        if mode != 'audit':
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


@APP.function(image=CPU_IMAGE, cpu=2, memory=4096, timeout=300, retries=0,
              volumes={'/datasets': DATA.read_only(), '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def label_audit(run_id: str):
    return data_job(run_id, 'audit')


YOLO_IMAGE = (modal.Image.from_dockerfile(HERE.parent / 'Dockerfile.yolo')
              .env({**ENV, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                    'OPENBLAS_NUM_THREADS': '4'})
              .add_local_dir(HERE.parent / 'patches', '/work/patches')
              .add_local_file(HERE / 'yolo_stage.py', '/work/yolo_stage.py')
              .add_local_file(HERE / 'yolo_train.py', '/work/yolo_train.py')
              .add_local_file(HERE / 'yolo_smoke.py', '/work/yolo_smoke.py')
              .add_local_file(HERE / 'yolo_modal.py', '/work/yolo_modal.py')
              .add_local_file(HERE.parent / 'Dockerfile.yolo', '/work/Dockerfile.yolo'))
MANIFEST_SHA = 'bdb2b2a87d2b93af07d977dafac14cf5f99c2e3333e97350cada17da8475356e'


def experiment_job(run_id: str, mode: str):
    import datetime
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import time
    output = Path('/outputs') / run_id
    output.mkdir(parents=True, exist_ok=False)
    clock = time.monotonic()
    ceiling = 840 if mode == 'smoke' else 1740
    record = {'run_id': run_id, 'mode': mode, 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'modal_app_id': APP.app_id, 'modal_function_call_id': modal.current_function_call_id(),
              'modal_task_id': os.environ.get('MODAL_TASK_ID'), 'max_wall_seconds': ceiling + 60,
              'manifest_sha256': MANIFEST_SHA}
    (output / 'started.json').write_text(json.dumps(record, indent=2) + '\n')
    OUTPUT.commit()
    code = 1
    try:
        sources = output / 'sources'
        sources.mkdir()
        for source in Path('/work').rglob('*'):
            if source.is_file():
                target = sources / source.relative_to('/work')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        for patch in sorted(Path('/work/patches').glob('yolo-*.patch')):
            subprocess.run(['git', '-C', '/opt/yolov5', 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', '/opt/yolov5', 'apply', str(patch)], check=True)
        with (output / 'applied-upstream.patch').open('w') as stream:
            subprocess.run(['git', '-C', '/opt/yolov5', 'diff'], stdout=stream, check=True)
        with (output / 'freeze.txt').open('w') as stream:
            subprocess.run(['python', '-m', 'pip', 'freeze'], stdout=stream, check=True)
        with (output / 'hardware.txt').open('w') as stream:
            subprocess.run(['uname', '-a'], stdout=stream, check=True)
            if mode != 'smoke':
                subprocess.run(['nvidia-smi'], stdout=stream, check=True)
        commands = [['python', '/work/yolo_stage.py', '--manifest-sha256', MANIFEST_SHA, '--preflight']]
        if mode == 'smoke':
            commands.append(['python', '/work/yolo_smoke.py', str(output / 'smoke.json')])
        else:
            for model in (['s'] if mode == 'recovery' else ['s', 'm', 'l']):
                command = ['python', '/work/yolo_train.py', '--model', model, '--data', '/tmp/yolo-data/data.yaml',
                           '--output', str(output / model), '--preflight']
                commands.append(command)
                if model == 's':
                    commands.append(command + ['--resume-check'])
                    if mode == 'recovery':
                        metrics = str(output / model / 'metrics.json')
                        reference = str(output / model / 'metrics-before-recovery.json')
                        commands.append(['python', '-c', 'import pathlib,sys;pathlib.Path(sys.argv[1]).rename(sys.argv[2])', metrics, reference])
                        commands.append(command + ['--resume'])
                        commands.append(['python', '-c',
                            'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); keys=["native_test_metrics","common_pycocotools_stats","selected_model_state_sha256"]; assert all(a[k]==b[k] for k in keys); print("Evaluation recovery metrics and checkpoint exact")',
                            metrics, reference])
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (output / 'console.log').open('w') as stream:
            for command in commands:
                stream.write(json.dumps({'command': command}) + '\n')
                stream.flush()
                remaining = ceiling - (time.monotonic() - clock)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, ceiling)
                code = subprocess.run(command, cwd='/opt/yolov5', stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=remaining).returncode
                OUTPUT.commit()
                if code:
                    break
    except subprocess.TimeoutExpired:
        code = 124
        raise
    finally:
        record.update({'finished_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'wall_seconds': time.monotonic() - clock, 'exit_code': code})
        def digest(path):
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        record['artifacts'] = {str(path.relative_to(output)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                               for path in output.rglob('*') if path.is_file() and path.name != 'execution.json'}
        (output / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        OUTPUT.commit()
    if code:
        raise RuntimeError(f'{mode} exited {code}; see retained console')
    return json.dumps(record)


@APP.function(image=YOLO_IMAGE, cpu=4, memory=16384, timeout=900, retries=0,
              volumes={'/datasets': DATA.read_only(), '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def smoke(run_id: str):
    return experiment_job(run_id, 'smoke')


@APP.function(image=YOLO_IMAGE, gpu='A100-80GB', cpu=4, memory=65536, timeout=1800, retries=0,
              volumes={'/datasets': DATA.read_only(), '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def preflight(run_id: str, recovery_only: bool = False):
    return experiment_job(run_id, 'recovery' if recovery_only else 'preflight')


@APP.local_entrypoint()
def main(run_id: str = 'data-acquisition-v1', mode: str = 'acquire'):
    functions = {'acquire': acquire, 'prepare': prepare, 'audit': label_audit,
                 'smoke': smoke, 'preflight': preflight}
    if mode == 'recovery':
        print(preflight.remote(run_id, True))
    else:
        assert mode in functions
        print(functions[mode].remote(run_id))
