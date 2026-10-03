"""One bounded CPU audit of existing internal author archives; no annotation changes."""
from pathlib import Path
import modal

APP = modal.App('repro-992e7a-lineage')
DATA = modal.Volume.from_name('datasets', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
OUTPUT = modal.Volume.from_name('repro-992e7a-results', version=2)
HERE = Path(__file__).parent
IMAGE = (modal.Image.debian_slim(python_version='3.11').pip_install('PyYAML==6.0.2')
         .env({'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub'})
         .add_local_file(HERE / 'lineage_audit.py', '/work/lineage_audit.py')
         .add_local_file(HERE / 'lineage_modal.py', '/work/lineage_modal.py'))


@APP.function(image=IMAGE, cpu=2, memory=4096, timeout=300, retries=0,
              volumes={'/datasets': DATA.with_mount_options(read_only=True),
                       '/cache/huggingface': CACHE, '/outputs': OUTPUT})
def audit():
    import datetime
    import hashlib
    import json
    import subprocess
    import time
    directory = Path('/outputs/cross-release-label-audit-v1')
    directory.mkdir(exist_ok=False)
    command = ['python', '/work/lineage_audit.py', str(directory / 'report.json')]
    started = time.monotonic()
    record = {'run_id': directory.name, 'modal_app_id': APP.app_id,
              'modal_function_call_id': modal.current_function_call_id(),
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'cpu': 2, 'memory_mib': 4096, 'gpu_count': 0,
              'timeout_seconds': 300, 'native_timeout_seconds': 270,
              'retry_ceiling': 0, 'cost_ceiling_chf': .25, 'command': command}
    for name in ('lineage_audit.py', 'lineage_modal.py'):
        (directory / name).write_bytes(Path('/work', name).read_bytes())
    (directory / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    OUTPUT.commit()
    code = 1
    try:
        with (directory / 'console.log').open('w') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=270).returncode
            except subprocess.TimeoutExpired:
                code = 124
    finally:
        record.update(exit_code=code, wall_seconds=time.monotonic() - started,
                      finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (directory / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        (directory / 'pip-freeze.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
        files = [{'path': p.name, 'bytes': p.stat().st_size,
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(directory.iterdir()) if p.is_file()]
        (directory / 'evidence.json').write_text(json.dumps(files, indent=2) + '\n')
        OUTPUT.commit()
    print((directory / 'console.log').read_text())
    print(json.dumps(record))
    return record


@APP.local_entrypoint()
def main():
    result = audit.remote()
    if result['exit_code']:
        raise SystemExit(result['exit_code'])
