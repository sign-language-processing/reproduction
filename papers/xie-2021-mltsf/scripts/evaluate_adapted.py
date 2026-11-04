"""Authorized full evaluation with per-video BatchNorm; no weight updates."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

import torch

import experiment as ex
from diagnose_bn import CHECKPOINT_ROOT, CHECKPOINTS, control_test, set_normalization
from model import MLTSF, collapse


@torch.no_grad()
def run(out, wall_seconds):
    start = time.monotonic()
    control_test()
    ex.seed_all()
    filename, checksum = CHECKPOINTS['best52']
    path = CHECKPOINT_ROOT / filename
    assert ex.sha(path) == checksum
    state = torch.load(path, map_location='cpu', weights_only=False)
    ex.validate_resume_recipe(state)
    assert state['epoch'] == 52
    vocabulary = state['vocabulary']
    model = MLTSF(len(vocabulary) + 1).cuda()
    model.load_state_dict(state['model'])
    set_normalization(model, 'video_all')
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    environment = {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                       '--format=csv,noheader'], text=True).strip(),
        'source_sha256': {p.name: ex.sha(p) for p in Path(__file__).parent.glob('*.py')}}
    ex.write_json(out / 'environment.json', environment)
    (out / 'pip-freeze.txt').write_text(subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True))
    report = {'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'conditional': True, 'variant': 'per-video-all-batchnorm',
        'checkpoint': {'epoch': 52, 'sha256': checksum, 'path': str(path)},
        'selection': 'Original earliest minimum-dev checkpoint, fixed before adapted evaluation; no reselection.',
        'protocol': 'Full dev and test; unchanged center crop, RGB scaling, greedy decoding and pinned scorer. '
                    'All BatchNorm layers use current-video statistics without updating running buffers. '
                    'No label-dependent adaptation, gradients, optimizer or weight updates.',
        'dev': None, 'test': None}
    previous = json.loads(Path('/results/bn-generalization-diagnostic-2/attempts/'
                               'bn-generalization-diagnostic-2/best52/dev/video_all/predictions.json').read_text())
    for split in ('dev', 'test'):
        rows = ex.rows(split)
        paths = {r['id']: ex.frame_paths(r, split) for r in rows}
        predictions = {}
        with ex.VideoLoader() as loader:
            for number, (row, (video, _)) in enumerate(loader.iterate(rows, paths), 1):
                if time.monotonic() - start > wall_seconds - 100:
                    raise TimeoutError('Adapted evaluation wall limit')
                logits, _ = model(video.cuda())
                assert torch.isfinite(logits).all()
                predictions[row['id']] = [vocabulary[i - 1] for i in collapse(logits.argmax(-1).tolist())]
                if number % 50 == 0:
                    progress = {'split': split, 'samples_done': number, 'samples_total': len(rows),
                                'elapsed_seconds': time.monotonic() - start}
                    ex.write_json(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        assert set(predictions) == {r['id'] for r in rows}
        if split == 'dev':
            matches = sum(predictions[k] == v for k, v in previous.items())
            assert matches == len(previous) == 128, 'Adapted predictions differ from validated diagnostic'
            report['diagnostic_dev_prediction_matches'] = matches
        report[split] = ex.score(predictions, split, out / split)
        assert report[split]['samples'] == ex.COUNTS[split]
        ex.write_json(out / 'result.json', report)
    assert all(torch.equal(v.cpu(), state['model'][k]) for k, v in model.state_dict().items())
    assert ex.sha(path) == checksum
    report.update(completed=True, parameters_and_buffers_unchanged=True,
                  finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - start)
    ex.write_json(out / 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['adapted_evaluation'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=1500)
    args = parser.parse_args()
    run(args.out, args.wall_seconds)
