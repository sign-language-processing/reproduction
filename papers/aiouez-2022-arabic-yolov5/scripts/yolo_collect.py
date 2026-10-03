"""CPU-only closed artifact collection after a bounded YOLO full run terminates."""
from pathlib import Path
import modal
APP = modal.App('repro-992e7a-yolo-collect')
OUTPUT = modal.Volume.from_name('repro-992e7a-results', version=2)
CACHE = modal.Volume.from_name('huggingface-cache', version=2)
IMAGE = (modal.Image.debian_slim(python_version='3.12')
         .env({'HF_HOME': '/cache/huggingface', 'HF_HUB_CACHE': '/cache/huggingface/hub'})
         .add_local_file(Path(__file__), '/work/yolo_collect.py'))


@APP.function(image=IMAGE, cpu=2, memory=4096, timeout=900, retries=0,
              volumes={'/outputs': OUTPUT, '/cache/huggingface': CACHE})
def collect(run_id: str):
    import csv
    import datetime
    import hashlib
    import json
    import re
    import time
    from concurrent.futures import ThreadPoolExecutor
    assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', run_id)
    output = Path('/outputs') / run_id
    execution = json.loads((output / 'execution.json').read_text())
    assert execution['exit_code'] is not None
    assert all(s['state'] != 'running' for s in execution['segments'])
    start = datetime.datetime.now(datetime.timezone.utc)
    clock = time.monotonic()
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return {'path': str(path.relative_to(output)), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
    receipt = {'started_at_utc': start.isoformat(), 'modal_app_id': APP.app_id,
               'modal_function_call_id': modal.current_function_call_id(),
               'collector_sha256': hashlib.sha256(Path('/work/yolo_collect.py').read_bytes()).hexdigest()}
    if execution['exit_code'] == 0:
        model = execution['model']
        metrics = json.loads((output / model / 'metrics.json').read_text())
        assert not metrics['preflight']
        assert digest(output / model / 'train/weights/best.pt')['sha256'] == metrics['best_checkpoint_sha256']
        assert len(metrics['matched_batch1_inference_seconds']['samples']) == 1509
        with (output / model / 'train/results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == (60 if model == 's' else 50)
        assert all(int(row[next(k for k in row if k.strip() == 'epoch')]) == i for i, row in enumerate(rows))
        receipt['validated_epochs'] = len(rows)
        receipt['validated_test_timing_count'] = 1509
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'evidence.json' and not p.name.startswith('collection-'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(digest, paths))
    (output / 'evidence.json').write_text(json.dumps(files, indent=2) + '\n')
    receipt.update(file_count=len(files), manifest_sha256=hashlib.sha256((output / 'evidence.json').read_bytes()).hexdigest(),
                   finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), wall_seconds=time.monotonic()-clock, exit_code=0)
    (output / ('collection-' + start.strftime('%Y%m%dT%H%M%S') + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    OUTPUT.commit()
    return receipt


@APP.local_entrypoint()
def main(run_id: str):
    print(collect.remote(run_id))
