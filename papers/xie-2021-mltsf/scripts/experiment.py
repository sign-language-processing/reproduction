"""Data audit, conditional training, preflight and pinned PHOENIX evaluation."""
import argparse
import csv
from contextlib import closing
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from model import MLTSF, collapse, forced_alignment, mapped_alignment, gfe_loss

ROOT = Path('/datasets/rwth-phoenix-2014/raw/phoenix2014-release/phoenix-2014-multisigner')
EVAL = Path('/opt/CorrNet/evaluation/slr_eval')
COUNTS = {'train': 5672, 'dev': 540, 'test': 629}
# Pinned CorrNet preprocessing drops CSV row2390 as broken. Exclusion approved
# on2026-11-02 after the same clip caused an impossible CTC path in our run.
BROKEN_TRAIN_ID = '13April_2011_Wednesday_tagesschau_default-14'
PRE_EXCLUSION_RECIPE = {
    'model.py': 'ef584d493f313eb4f2c489c63b02a7cd2d057e6324cbf6d7a0e0720c465ca06c',
    'experiment.py': '82f767b1a23298da7636bab59fc312249928ee1e976c2f983586340030cead52'}
HASHES = {'train': 'ebd3a83d0bb81b153e4c2acf2e6fcbc1ba6aa8211176017154a7213629a19947',
          'dev': '42ba319d167b96c90832bb90cf5da9059042ebc826e400bba4f6674a5d840cef',
          'test': 'e8f05c6573d9546cd9f5b8528663b4924930800bc34b7f2fcc15708316166990'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def rows(split):
    path = ROOT / 'annotations/manual' / f'{split}.corpus.csv'
    if sha(path) != HASHES[split]:
        raise ValueError(f'{split} annotations changed')
    result = list(csv.DictReader(path.open(), delimiter='|'))
    assert len(result) == COUNTS[split] == len({r['id'] for r in result})
    if split == 'train':
        assert result[2390]['id'] == BROKEN_TRAIN_ID
        result = [r for r in result if r['id'] != BROKEN_TRAIN_ID]
    return result


def frame_paths(row, split):
    base = ROOT / 'features/fullFrame-210x260px' / split
    paths = sorted(glob.glob(str(base / row['folder'])))
    if not paths or any(not p.endswith('.png') for p in paths):
        raise ValueError(f'Invalid frame layout for {split}/{row["id"]}')
    return paths


def minimum_ctc_frames(row):
    labels = row['annotation'].split()
    return max(16, 4 * (len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))))


def video_plan(n, augment=False, rng=random, min_frames=16):
    if augment and max(16, round(1.2*n)) < min_frames:
        raise ValueError('No CTC-valid temporal augmentation exists in the approved range')
    size = max(16, round(n * rng.uniform(.8, 1.2))) if augment else n
    # Approved conditional rejection sampling; no clip is removed by this guard.
    while augment and size < min_frames:
        size = max(16, round(n * rng.uniform(.8, 1.2)))
    indices = np.rint(np.linspace(0, n - 1, size)).astype(np.int64)
    left, top = (rng.randrange(33), rng.randrange(33)) if augment else (16, 16)
    return indices, left, top


def load_video(paths, augment=False, *, plan=None, pool=None, min_frames=16):
    indices, left, top = video_plan(len(paths), augment, min_frames=min_frames) if plan is None else plan
    # Shared crop and RGB [0,1] are explicit reconstruction choices.
    def decode(i):
        with Image.open(paths[i]) as im:
            im = im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
            return np.array(im.crop((left, top, left + 224, top + 224)))
    # Decode repeated temporal samples once, retaining the original pixel values.
    unique = list(dict.fromkeys(indices))
    decoded = dict(zip(unique, map(decode, unique) if pool is None else pool.map(decode, unique)))
    frames = [decoded[i] for i in indices]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255)
    return video, torch.from_numpy(indices)


class VideoLoader:
    """Bounded ordered prefetch; only consumed videos advance augmentation RNG."""
    def __init__(self, workers=8):
        self.frames = ThreadPoolExecutor(max_workers=workers)
        self.videos = ThreadPoolExecutor(max_workers=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.videos.shutdown(wait=True, cancel_futures=True)
        self.frames.shutdown(wait=True, cancel_futures=True)

    def iterate(self, items, paths, augment=False):
        rng = random.Random()
        rng.setstate(random.getstate())
        items = iter(items)
        pending = deque()
        def submit():
            row = next(items, None)
            if row is not None:
                files = paths[row['id']]
                plan = video_plan(len(files), augment, rng, minimum_ctc_frames(row) if augment else 16)
                pending.append((row, rng.getstate(), self.videos.submit(
                    load_video, files, plan=plan, pool=self.frames)))
        try:
            submit()
            submit()
            while pending:
                row, state, future = pending.popleft()
                video = future.result()
                if augment:
                    random.setstate(state)
                yield row, video
                submit()
        finally:
            for _, _, future in pending:
                future.cancel()


def audit(out):
    report = {'splits': {}, 'vocabulary': None, 'frame_inventory_sha256': None}
    inventory = hashlib.sha256()
    all_ids = set()
    vocab = set()
    for split in COUNTS:
        lengths = []
        references = {line.split()[0] for line in (EVAL / f'phoenix2014-groundtruth-{split}.stm').read_text().splitlines()}
        rs = rows(split)
        if split != 'train':
            assert references == {r['id'] for r in rs}, f'{split}: scorer reference IDs differ'
        assert not all_ids.intersection(r['id'] for r in rs), 'Split leakage by ID'
        all_ids.update(r['id'] for r in rs)
        with ThreadPoolExecutor(max_workers=32) as pool:
            inventories = list(pool.map(lambda row: frame_paths(row, split), rs))
        for row, paths in zip(rs, inventories):
            lengths.append((len(paths), row['id']))
            for path in paths:
                rel = str(Path(path).relative_to(ROOT))
                inventory.update(f'{rel}\n'.encode())
            if split == 'train':
                vocab.update(row['annotation'].split())
        lengths.sort()
        report['splits'][split] = {'samples': len(rs), 'unused_train_scorer_missing_ids': len({r['id'] for r in rs} - references) if split == 'train' else 0, 'frames': sum(n for n, _ in lengths),
            'min': lengths[0], 'median': lengths[len(lengths)//2],
            'p95': lengths[int(.95*len(lengths))], 'max': lengths[-1],
            'csv_sha256': HASHES[split], 'scorer_stm_sha256': sha(EVAL / f'phoenix2014-groundtruth-{split}.stm')}
        print(json.dumps({split: report['splits'][split]}), flush=True)
    report['vocabulary'] = sorted(vocab)
    report['frame_inventory_sha256'] = inventory.hexdigest()
    report['inventory_basis'] = 'Sorted per-sample frame paths; annotations and release manifest are separately hashed, not all pixel contents'
    write_json(out / 'data-audit.json', report)
    return report


def score(predictions, split, out):
    """Run inspected upstream normalization/merge and NIST sclite, without os.system."""
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'predictions.json', predictions)
    ids = set(predictions)
    stm = [x for x in (EVAL / f'phoenix2014-groundtruth-{split}.stm').read_text().splitlines()
           if x.split()[0] in ids]
    assert len(stm) == len(ids), 'Reference coverage mismatch'
    (out / 'reference.stm').write_text('\n'.join(sorted(stm)) + '\n')
    # Only timestamps for ordering; utterance scoring is controlled by STM.
    ctm = []
    for uid, words in sorted(predictions.items()):
        for i, word in enumerate(words or ['[EMPTY]']):
            ctm.append(f'{uid} 1 {i*.03:.3f} 0.030 {word}')
    (out / 'hypothesis.ctm').write_text('\n'.join(ctm) + '\n')
    subprocess.run(['bash', str(EVAL / 'preprocess.sh'), str(out / 'hypothesis.ctm'),
                    str(out / 'temporary.ctm'), str(out / 'normalized.ctm')], check=True)
    subprocess.run(['python', str(EVAL / 'mergectmstm.py'), str(out / 'normalized.ctm'),
                    str(out / 'reference.stm')], check=True)
    command = ['sclite', '-h', str(out / 'normalized.ctm'), 'ctm', '-r',
               str(out / 'reference.stm'), 'stm', '-f', '0', '-o', 'dtl', 'stdout']
    text = subprocess.check_output(command, text=True)
    (out / 'sclite.dtl').write_text(text)
    print(text, flush=True)
    rates = {}
    for key, label in [('wer', 'Percent Total Error'), ('del', 'Percent Deletions'), ('ins', 'Percent Insertions')]:
        match = re.search(label + r'\s*=\s*([\d.]+)%', text)
        if not match:
            raise ValueError(f'Unparsed sclite field {label}; native report retained')
        rates[key] = float(match.group(1))
    rates.update(samples=len(predictions), scorer_sha256=sha(out / 'sclite.dtl'))
    write_json(out / 'metrics.json', rates)
    return rates


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_checkpoint(path, model, optimizer, epoch, best, vocabulary, alignments, progress=None):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'recipe_sha256': {f: sha(Path(__file__).with_name(f)) for f in ['model.py', 'experiment.py']},
             'best': best, 'vocabulary': vocabulary, 'alignments': alignments, 'progress': progress,
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def validate_resume_recipe(state):
    current = {f: sha(Path(__file__).with_name(f)) for f in ['model.py', 'experiment.py']}
    if state['recipe_sha256'] == current:
        return
    assert state['recipe_sha256'] == PRE_EXCLUSION_RECIPE, 'Resume recipe changed'
    progress = state.get('progress')
    assert state['epoch'] == 0 and progress and progress['phase'] == 'train', 'Unsupported legacy checkpoint phase'
    order = progress['order']
    assert len(order) == len(set(order)) == 5672 and order.count(BROKEN_TRAIN_ID) == 1
    assert BROKEN_TRAIN_ID not in order[:progress['next_index']], 'Broken clip already processed'
    assert not state['alignments'], 'Unexpected legacy GFE cache'
    # Both retained checkpoints predate this clip. Preserve completed updates,
    # optimizer and RNG; remove only the unconsumed broken item from the order.
    progress['order'] = [uid for uid in order if uid != BROKEN_TRAIN_ID]
    state['resume_migration'] = {'old_recipe_sha256': state['recipe_sha256'],
        'new_recipe_sha256': current, 'excluded_training_id': BROKEN_TRAIN_ID,
        'next_index': progress['next_index'], 'completed_updates_preserved': True,
        'augmentation_guard': 'Redraw only CTC-infeasible scales within[0.8,1.2], approved2026-11-02'}


def restore(path, model, optimizer):
    # Only checkpoints generated by this entry point are loaded.
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_resume_recipe(state)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


@torch.no_grad()
def evaluate(model, rs, split, vocabulary, out):
    model.eval()
    predictions = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        paths = dict(zip([r['id'] for r in rs], pool.map(lambda r: frame_paths(r, split), rs)))
    with VideoLoader() as loader:
        for row, (x, _) in loader.iterate(rs, paths):
            logits, _ = model(x.to(next(model.parameters()).device))
            predictions[row['id']] = [vocabulary[i - 1] for i in collapse(logits.argmax(-1).tolist())]
    return score(predictions, split, out)


def train_step(model, optimizer, row, vocabulary, paths, canonical=None, prepared=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, indices = load_video(paths, augment=True, min_frames=minimum_ctc_frames(row)) if prepared is None else prepared
    device = next(model.parameters()).device
    target = torch.tensor([vocabulary.index(w) + 1 for w in row['annotation'].split()], dtype=torch.long)
    logits, auxiliary = model(x.to(device))
    # Mean negative sequence log probability for batch size 1, not target-length normalization.
    loss = F.ctc_loss(logits.log_softmax(-1)[:, None], target.to(device), [len(logits)], [len(target)], reduction='sum')
    if canonical is not None:
        mapped = mapped_alignment(canonical, indices, len(logits)).to(device)
        loss = loss + .05 * gfe_loss(auxiliary, mapped)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite CTC/GFE loss')
    loss.backward()
    optimizer.step()
    return float(loss.detach()), len(x)


def preflight(out, checkpoint_backbone):
    subprocess.run(['python', str(Path(__file__).with_name('test_model.py'))], check=True)
    seed_all()
    vocabulary = sorted({w for r in rows('train') for w in r['annotation'].split()})
    rs = rows('train')
    with ThreadPoolExecutor(max_workers=32) as pool:
        inventories = list(pool.map(lambda row: frame_paths(row, 'train'), rs))
    paths = {r['id']: files for r, files in zip(rs, inventories)}
    ordered = sorted(rs, key=lambda r: len(paths[r['id']]))
    chosen = [ordered[len(rs)//2], ordered[int(.95*len(rs))], ordered[-1]]
    result = {'steps': [], 'checkpoint_roundtrip': False}
    tiny_ids = {r['id'] for r in rows('dev')[:3]}
    perfect = {parts[0]: parts[5:] for line in (EVAL / 'phoenix2014-groundtruth-dev.stm').read_text().splitlines()
               if (parts := line.split())[0] in tiny_ids}
    result['scorer_self_check'] = score(perfect, 'dev', out / 'scorer-perfect')
    assert result['scorer_self_check']['wer'] == 0, 'Perfect-reference scorer self-check failed'
    model = MLTSF(len(vocabulary)+1, gfe=True, checkpoint_backbone=checkpoint_backbone).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for row in chosen:
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        # Create a real forced alignment on canonical input to exercise auxiliary loss.
        model.eval()
        with torch.no_grad():
            x, _ = load_video(paths[row['id']])
            logits, _ = model(x.cuda())
            labels = [vocabulary.index(w)+1 for w in row['annotation'].split()]
            canonical = forced_alignment(logits.log_softmax(-1), labels)
            del x, logits
        alignment_seconds = time.monotonic() - start
        step_start = time.monotonic()
        loss, frames = train_step(model, optimizer, row, vocabulary, paths[row['id']], canonical)
        torch.cuda.synchronize()
        entry = {'frames': frames, 'loss': loss, 'seconds_including_alignment': time.monotonic()-start,
                 'alignment_seconds': alignment_seconds, 'training_seconds': time.monotonic()-step_start,
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        result['steps'].append(entry)
        write_json(out / 'preflight.json', result)
        print(json.dumps(entry), flush=True)
    save_checkpoint(out / 'roundtrip.pt', model, optimizer, 0, float('inf'), vocabulary, {})
    # Check model outputs AND RNG recovery after save/reload.
    model.eval()
    x, _ = load_video(paths[chosen[0]['id']])
    with torch.no_grad():
        expected = model(x.cuda())[0].cpu()
    state = restore(out / 'roundtrip.pt', model, optimizer)
    with torch.no_grad():
        actual = model(x.cuda())[0].cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_rng = torch.rand(5, device='cuda')
    restore(out / 'roundtrip.pt', model, optimizer)
    assert torch.equal(expected_rng, torch.rand(5, device='cuda'))
    # Verify optimizer can take another finite step after restore, then evaluate.
    train_step(model, optimizer, chosen[0], vocabulary, paths[chosen[0]['id']])
    result['checkpoint_roundtrip'] = True
    result['evaluation'] = evaluate(model, rows('dev')[:3], 'dev', vocabulary, out / 'dev-tiny')
    result['parameters'] = sum(p.numel() for p in model.parameters())
    write_json(out / 'preflight.json', result)
    return result


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def train(out, gfe, checkpoint_backbone, wall_seconds, *, device='cuda', epochs=60, checkpoint_every=100):
    seed_all()
    vocabulary = sorted({w for r in rows('train') for w in r['annotation'].split()})
    model = MLTSF(len(vocabulary)+1, gfe, checkpoint_backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rs = rows('train')
    with ThreadPoolExecutor(max_workers=32) as pool:
        inventories = list(pool.map(lambda row: frame_paths(row, 'train'), rs))
    paths = {r['id']: files for r, files in zip(rs, inventories)}
    by_id = {r['id']: r for r in rs}
    start_epoch, best, alignments, progress = 1, float('inf'), {}, None
    last = out / 'last.pt'
    if last.exists():
        state = restore(last, model, optimizer)
        assert state['vocabulary'] == vocabulary
        start_epoch, best, alignments = state['epoch']+1, state['best'], state['alignments']
        progress = state.get('progress')
        if 'resume_migration' in state:
            assert set(progress['order']) == set(by_id), 'Migrated order differs from approved training split'
            write_json(out / 'resume-migration.json', state['resume_migration'])
    started = time.monotonic()
    saved_at = started

    def checkpoint(epoch):
        nonlocal saved_at
        save_checkpoint(last, model, optimizer, epoch-1, best, vocabulary, alignments, progress)
        saved_at = time.monotonic()
        write_json(out / 'progress.json', {'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'epoch': epoch, 'phase': progress['phase'], 'next_index': progress['next_index'],
            'best_dev_wer': best if best != float('inf') else None,
            'segment_elapsed_seconds': saved_at-started})

    def stop(epoch):
        if STOP_REQUESTED or time.monotonic()-started >= wall_seconds:
            checkpoint(epoch)
            write_json(out / 'pause.json', {'epoch': epoch, 'phase': progress['phase'],
                'next_index': progress['next_index'], 'reason': 'Graceful wall limit or termination signal'})
            return True
        return False

    with VideoLoader() as loader:
        for epoch in range(start_epoch, epochs+1):
            for group in optimizer.param_groups:
                group['lr'] = 1e-4 * .5**sum(epoch >= milestone for milestone in [45, 50, 55])
            if progress is None:
                progress = {'phase': 'alignment' if gfe and epoch in [16, 26, 36, 46, 56] else 'shuffle',
                            'next_index': 0, 'total_loss': 0.}
                if progress['phase'] == 'alignment':
                    alignments = {}
            if stop(epoch):
                return
            if progress['phase'] == 'alignment':
                model.eval()
                with torch.no_grad(), closing(loader.iterate(rs[progress['next_index']:], paths)) as videos:
                    while progress['next_index'] < len(rs):
                        if stop(epoch):
                            return
                        row, (x, _) = next(videos)
                        logits, _ = model(x.to(device))
                        labels = [vocabulary.index(w)+1 for w in row['annotation'].split()]
                        alignments[row['id']] = forced_alignment(logits.log_softmax(-1), labels)
                        progress['next_index'] += 1
                        if progress['next_index'] % checkpoint_every == 0 or time.monotonic()-saved_at >= 300:
                            checkpoint(epoch)
                progress.update(phase='shuffle', next_index=0)
            if progress['phase'] == 'shuffle':
                order = [r['id'] for r in rs]
                random.shuffle(order)
                progress.update(phase='train', order=order, next_index=0)
                checkpoint(epoch)
            if progress['phase'] == 'train':
                order = [by_id[uid] for uid in progress['order'][progress['next_index']:]]
                with closing(loader.iterate(order, paths, augment=True)) as videos:
                    while progress['next_index'] < len(rs):
                        if stop(epoch):
                            return
                        row, prepared = next(videos)
                        loss, _ = train_step(model, optimizer, row, vocabulary, paths[row['id']],
                                             alignments.get(row['id']), prepared=prepared)
                        progress['total_loss'] += loss
                        progress['next_index'] += 1
                        if progress['next_index'] % checkpoint_every == 0 or time.monotonic()-saved_at >= 300:
                            checkpoint(epoch)
                            print(json.dumps({'epoch': epoch, 'step': progress['next_index'], 'loss': loss}), flush=True)
                progress.update(phase='dev', next_index=0)
                checkpoint(epoch)
            if stop(epoch):
                return
            # A hard interruption during dev can repeat dev evaluation, never training.
            metrics = evaluate(model, rows('dev'), 'dev', vocabulary, out / f'dev-{epoch:02d}')
            improved = metrics['wer'] < best
            best = min(best, metrics['wer'])
            if improved:
                save_checkpoint(out / 'best.pt', model, optimizer, epoch, best, vocabulary, alignments)
            with (out / 'epochs.jsonl').open('a') as f:
                f.write(json.dumps({'epoch': epoch, 'loss': progress['total_loss']/len(rs),
                                    'dev': metrics, 'selected': improved})+'\n')
            progress = None
            save_checkpoint(last, model, optimizer, epoch, best, vocabulary, alignments)
            write_json(out / 'progress.json', {'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'epoch': epoch, 'phase': 'epoch_complete', 'next_index': len(rs),
                'best_dev_wer': best, 'dev': metrics, 'segment_elapsed_seconds': time.monotonic()-started})
    restore(out / 'best.pt', model, optimizer)
    test = evaluate(model, rows('test'), 'test', vocabulary, out / 'test-selected')
    write_json(out / 'result.json', {'best_dev_wer': best, 'test': test, 'conditional': True})
    (out / 'pause.json').unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['audit', 'preflight', 'train'])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--gfe', action='store_true')
    p.add_argument('--checkpoint-backbone', action='store_true')
    p.add_argument('--wall-seconds', type=int, default=82800)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    write_json(a.out / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'gpu': subprocess.getoutput('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'),
        'image_id': os.environ.get('MODAL_IMAGE_ID'), 'mode': a.mode, 'gfe': a.gfe or a.mode == 'preflight',
        'checkpoint_backbone': a.checkpoint_backbone,
        'source_sha256': {f.name: sha(f) for f in Path(__file__).parent.glob('*.py')}})
    (a.out / 'pip-freeze.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
    if a.mode == 'audit':
        audit(a.out)
    elif a.mode == 'preflight':
        preflight(a.out, a.checkpoint_backbone)
    else:
        signal.signal(signal.SIGTERM, request_stop)
        train(a.out, a.gfe, a.checkpoint_backbone, a.wall_seconds)


if __name__ == '__main__':
    main()
