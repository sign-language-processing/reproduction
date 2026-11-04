"""Bounded Modal jobs. Always invoke via modal_repro_sign.sh."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import shutil
import re
import threading
import time
import modal

PAPER = Path(__file__).resolve().parents[1]
# Reuse the successfully built immutable environment; upload only changed code.
# REBUILD_IMAGE=1 rebuilds the equivalent environment from the pinned Dockerfile.
IMAGE = (modal.Image.from_dockerfile(PAPER / 'Dockerfile', context_dir=PAPER)
         if os.environ.get('REBUILD_IMAGE') == '1' else modal.Image.from_id('im-r8xSTWgJSfiL0bcEBgGYBJ'))
IMAGE = IMAGE.add_local_dir(PAPER / 'scripts', '/workspace/scripts', copy=True)
DATA = modal.Volume.from_name('datasets')
CACHE = modal.Volume.from_name('huggingface-cache')
RESULTS = modal.Volume.from_name('xie-a514cc97-results', create_if_missing=True, version=2)
APP = modal.App('repro-xie-a514cc97')
MOUNTS = {'/datasets': DATA.read_only(), '/cache/huggingface': CACHE, '/results': RESULTS}


def execute(mode, run_id, gfe=False, checkpoint_backbone=False, wall_seconds=1800, attempt_id=None):
    out = Path('/results') / run_id
    out.mkdir(parents=True, exist_ok=True)
    evidence = out
    if attempt_id is not None:
        if not re.fullmatch(r'[a-z0-9-]+', attempt_id):
            raise ValueError('Invalid attempt ID')
        evidence = out / 'attempts' / attempt_id
        # Never overwrite terminal or currently running attempt evidence.
        evidence.mkdir(parents=True, exist_ok=False)
    entrypoint = {'benchmark': 'benchmark.py', 'bn_diagnostic': 'diagnose_bn.py',
                  'adapted_evaluation': 'evaluate_adapted.py'}.get(mode, 'experiment.py')
    command = ['python', '-u', '/workspace/scripts/' + entrypoint, mode, '--out', str(out),
               '--wall-seconds', str(wall_seconds)]
    if gfe:
        command.append('--gfe')
    if checkpoint_backbone:
        command.append('--checkpoint-backbone')
    stop = threading.Event()
    def persist():
        while not stop.wait(60):
            RESULTS.commit()
    thread = threading.Thread(target=persist, daemon=True)
    thread.start()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    begin = time.monotonic()
    code = -1
    (evidence / 'started.json').write_text(json.dumps({'started_at_utc': started,
        'function_call_id': modal.current_function_call_id(),
        'wall_seconds': wall_seconds, 'run_id': run_id, 'attempt_id': attempt_id,
        'command': command, 'task_id': os.environ.get('MODAL_TASK_ID')}, indent=2)+'\n')
    RESULTS.commit()
    try:
        with (evidence / 'stdout.log').open('a') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            deadline = threading.Timer(wall_seconds, lambda: process.send_signal(signal.SIGTERM))
            hard_deadline = threading.Timer(wall_seconds + 120, process.kill)
            deadline.start()
            hard_deadline.start()
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            deadline.cancel()
            hard_deadline.cancel()
    finally:
        stop.set()
        thread.join()
        summary = {'command': command, 'exit_code': code, 'started_at_utc': started,
            'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'seconds': time.monotonic()-begin, 'function_call_id': modal.current_function_call_id(),
            'task_id': os.environ.get('MODAL_TASK_ID'), 'image_id': os.environ.get('MODAL_IMAGE_ID')}
        summary['training_completed'] = (out / 'result.json').exists() if mode == 'train' else None
        if attempt_id is not None:
            # Mutable checkpoints stay at the run root for resume; each attempt
            # gets an immutable snapshot before another segment can replace them.
            for p in list(out.rglob('*')):
                relative = p.relative_to(out)
                if p.is_file() and relative.parts[0] != 'attempts':
                    dest = evidence / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest)
        (evidence / 'run.json').write_text(json.dumps(summary, indent=2)+'\n')
        manifest = {}
        for p in evidence.rglob('*'):
            if p.is_file() and p.name != 'artifact-hashes.json':
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda: f.read(8 << 20), b''):
                        h.update(b)
                manifest[str(p.relative_to(evidence))] = h.hexdigest()
        (evidence / 'artifact-hashes.json').write_text(json.dumps(manifest, indent=2)+'\n')
        RESULTS.commit()
    return summary


@APP.function(image=IMAGE, cpu=8, memory=16384, volumes=MOUNTS, timeout=1800)
def audit(run_id: str):
    return execute('audit', run_id)


@APP.function(image=IMAGE, gpu='A100-40GB', cpu=8, memory=16384, volumes=MOUNTS, timeout=3600)
def preflight(run_id: str, checkpoint_backbone: bool = False):
    return execute('preflight', run_id, checkpoint_backbone=checkpoint_backbone)


@APP.function(image=IMAGE, gpu='A100-40GB', cpu=8, memory=16384, volumes=MOUNTS, timeout=2100)
def benchmark(run_id: str):
    return execute('benchmark', run_id, wall_seconds=1800)


@APP.function(image=IMAGE, gpu='A100-40GB', cpu=(8, 8), memory=(16384, 16384),
              volumes=MOUNTS, timeout=15900, startup_timeout=600,
              scaledown_window=2, max_containers=2, retries=0)
def train(run_id: str, attempt_id: str, gfe: bool = False,
          checkpoint_backbone: bool = False, wall_seconds: int = 14400):
    if not 1 <= wall_seconds <= 14400:
        raise ValueError('Training segment must be between 1 second and 4 hours')
    # Bound the whole invocation, including a stuck final snapshot/hash/commit.
    # GPU billing during startup is separately reserved in the shared budget.
    hard_stop = threading.Timer(wall_seconds + 300, os._exit, args=(124,))
    hard_stop.daemon = True
    hard_stop.start()
    try:
        return execute('train', run_id, gfe, checkpoint_backbone, wall_seconds, attempt_id)
    finally:
        hard_stop.cancel()


@APP.local_entrypoint()
def volumes():
    # Public info omits version; inspect metadata after normal synchronous hydrate.
    from modal._utils.async_utils import synchronizer
    for volume in (DATA, CACHE):
        volume.hydrate()
        version = synchronizer._translate_in(volume)._metadata.version
        print(json.dumps({'name': volume.info().name, 'version': version}))
        if version != 2:
            raise ValueError('Shared volume is not VolumeFS v2')


@APP.function(image=IMAGE, gpu='A100-40GB', cpu=(8, 8), memory=(16384, 16384),
              volumes=MOUNTS, timeout=1800, startup_timeout=600,
              scaledown_window=2, max_containers=1, retries=0)
def diagnose_bn(run_id: str):
    return execute('bn_diagnostic', run_id, wall_seconds=1500, attempt_id=run_id)


@APP.function(image=IMAGE, gpu='A100-40GB', cpu=(8, 8), memory=(16384, 16384),
              volumes=MOUNTS, timeout=1800, startup_timeout=600,
              scaledown_window=2, max_containers=1, retries=0)
def evaluate_adapted(run_id: str):
    return execute('adapted_evaluation', run_id, wall_seconds=1500, attempt_id=run_id)


@APP.function(image=IMAGE, cpu=(2, 2), memory=(4096, 4096), volumes=MOUNTS,
              timeout=300, startup_timeout=300, scaledown_window=2, retries=0)
def diagnose_ctc():
    """Read-only checkpoint/length forensics; no training or recipe changes."""
    import sys
    import torch
    sys.path.insert(0, '/workspace/scripts')
    import experiment as ex
    out = Path('/results/ctc-diagnostic-1')
    out.mkdir(exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    state = torch.load('/results/full-mltsf-seed42/last.pt', map_location='cpu', weights_only=False)
    by_id = {r['id']: r for r in ex.rows('train')}
    rng = ex.random.Random()
    rng.setstate(state['python_rng'])
    progress = state['progress']
    candidates = []
    for index, uid in enumerate(progress['order'][progress['next_index']:progress['next_index']+100], progress['next_index']):
        row = by_id[uid]
        files = ex.frame_paths(row, 'train')
        indices, _, _ = ex.video_plan(len(files), True, rng)
        labels = row['annotation'].split()
        required = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
        candidates.append({'order_index': index, 'id': uid, 'source_frames': len(files),
            'augmented_frames': len(indices), 'encoder_steps': len(indices)//4,
            'target_tokens': len(labels), 'required_ctc_steps': required,
            'ctc_path_possible': len(indices)//4 >= required})
    report = {'started_at_utc': started, 'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'function_call_id': modal.current_function_call_id(), 'task_id': os.environ.get('MODAL_TASK_ID'),
        'image_id': os.environ.get('MODAL_IMAGE_ID'), 'checkpoint_sha256': ex.sha('/results/full-mltsf-seed42/last.pt'),
        'progress': {k: progress[k] for k in ['phase', 'next_index', 'total_loss']},
        'model_finite': all(bool(torch.isfinite(v).all()) for v in state['model'].values()),
        'optimizer_finite': all(bool(torch.isfinite(v).all()) for s in state['optimizer']['state'].values()
                                for v in s.values() if isinstance(v, torch.Tensor)),
        'candidates': candidates}
    ex.write_json(out / 'diagnostic.json', report)
    ex.write_json(out / 'artifact-hashes.json', {'diagnostic.json': ex.sha(out / 'diagnostic.json')})
    RESULTS.commit()
    print(json.dumps(report), flush=True)


@APP.function(image=IMAGE, cpu=(2, 2), memory=(4096, 4096), volumes=MOUNTS,
              timeout=600, startup_timeout=300, scaledown_window=2, retries=0)
def audit_exclusion(run_id: str = 'ctc-exclusion-audit-2'):
    """Validate the approved exclusion and both checkpoints before GPU resume."""
    import sys
    import torch
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, '/workspace/scripts')
    import experiment as ex
    out = Path('/results') / run_id
    out.mkdir(exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    rs = ex.rows('train')
    assert len(rs) == 5671
    with ThreadPoolExecutor(max_workers=32) as pool:
        counts = list(pool.map(lambda row: len(ex.frame_paths(row, 'train')), rs))
    invalid = []
    guard_affected = []
    slacks = []
    for row, frames in zip(rs, counts):
        labels = row['annotation'].split()
        required = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
        steps = max(16, round(.8*frames))//4
        slacks.append(max(steps, required)-required)
        if steps < required:
            guard_affected.append({'id': row['id'], 'source_frames': frames,
                            'minimum_encoder_steps': steps, 'required_steps': required})
        if frames//4 < required or max(16, round(1.2*frames))//4 < required:
            invalid.append({'id': row['id'], 'source_frames': frames, 'required_steps': required})
        else:
            rng = ex.random.Random(42)
            for _ in range(100 if steps < required else 1):
                indices, _, _ = ex.video_plan(frames, True, rng, ex.minimum_ctc_frames(row))
                assert len(indices)//4 >= required
                assert max(16, round(.8*frames)) <= len(indices) <= max(16, round(1.2*frames))
    vocabulary = sorted({w for r in rs for w in r['annotation'].split()})
    checkpoints = {}
    for run_id in ('full-mltsf-seed42', 'full-mltsf-gfe-seed42'):
        path = Path('/results') / run_id / 'last.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['vocabulary'] == vocabulary
        assert all(bool(torch.isfinite(v).all()) for v in state['model'].values())
        ex.validate_resume_recipe(state)
        checkpoints[run_id] = {'sha256': ex.sha(path), 'next_index': state['progress']['next_index'],
            'migration': state.get('resume_migration'), 'model_finite': True}
        del state
    report = {'started_at_utc': started, 'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'function_call_id': modal.current_function_call_id(), 'task_id': os.environ.get('MODAL_TASK_ID'),
        'image_id': os.environ.get('MODAL_IMAGE_ID'), 'training_samples': len(rs),
        'training_frames': sum(counts), 'minimum_ctc_slack': min(slacks), 'invalid_samples': invalid,
        'augmentation_guard_affected_samples': guard_affected,
        'checkpoints': checkpoints, 'passed': not invalid,
        'source_sha256': {p.name: ex.sha(p) for p in Path('/workspace/scripts').glob('*.py')}}
    ex.write_json(out / 'audit.json', report)
    ex.write_json(out / 'artifact-hashes.json', {'audit.json': ex.sha(out / 'audit.json')})
    RESULTS.commit()
    print(json.dumps(report), flush=True)
    assert not invalid, 'Additional CTC-infeasible samples require investigation before GPU resume'
