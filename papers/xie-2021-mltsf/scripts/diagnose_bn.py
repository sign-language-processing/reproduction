"""Read-only checkpoint diagnostics, not a replacement evaluation protocol."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

import experiment as ex
from model import MLTSF, collapse


CHECKPOINT_ROOT = Path('/results/full-mltsf-seed42/attempts/full-mltsf-segment-5')
CHECKPOINTS = {
    'best52': ('best.pt', '1bda011c0c95c5e01bc1fd40d5c36e99be8f85e1d5817258421514d5e5c6dc44'),
    'last60': ('last.pt', 'e91d28531c477a5a7f5e8917c9833fc66fdae7c5ba9e438d387da24382ff89ec'),
}


def set_normalization(model, mode):
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            use_video = (mode == 'video_all' or
                         mode == 'video_spatial' and name.startswith('backbone.') or
                         mode == 'video_temporal' and name.startswith(('level1.', 'level2.')))
            # Training statistics without running-buffer/counter updates.
            module.training = use_video
            module.track_running_stats = not use_video


def control_test():
    torch.manual_seed(7)
    bn = nn.BatchNorm1d(3)
    x = torch.randn(1, 3, 20) + 4
    expected = F.batch_norm(x, None, None, bn.weight, bn.bias, True, 0., bn.eps)
    before = {k: v.clone() for k, v in bn.state_dict().items()}
    set_normalization(bn, 'video_all')
    torch.testing.assert_close(bn(x), expected)
    assert all(torch.equal(v, bn.state_dict()[k]) for k, v in before.items())
    set_normalization(bn, 'running')
    torch.testing.assert_close(bn(x), F.batch_norm(x, before['running_mean'], before['running_var'],
                                                bn.weight, bn.bias, False, 0., bn.eps))


def chosen_rows(split):
    # Same fixed hash-ranking rule for both splits, independent of model outputs.
    return sorted(ex.rows(split), key=lambda r: hashlib.sha256(
        ('bn-diagnostic-v1:' + r['id']).encode()).hexdigest())[:128]


@torch.no_grad()
def run(out, wall_seconds):
    start = time.monotonic()
    control_test()
    ex.seed_all()
    report = {'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'diagnostic_only': True, 'sample_rule': '128 smallest SHA256(bn-diagnostic-v1:ID) per split',
              'preprocessing': 'Unchanged canonical center crop; no augmentation; same inputs across modes',
              'control_test_passed': True, 'checkpoints': {}, 'results': {},
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'source_sha256': {p.name: ex.sha(p) for p in Path(__file__).parent.glob('*.py')}}
    selected = {split: chosen_rows(split) for split in ('train', 'dev')}
    report['sample_ids'] = {s: [r['id'] for r in rs] for s, rs in selected.items()}
    original_dev = json.loads((CHECKPOINT_ROOT / 'dev-52/predictions.json').read_text())
    for label, (filename, checksum) in CHECKPOINTS.items():
        path = CHECKPOINT_ROOT / filename
        assert ex.sha(path) == checksum
        state = torch.load(path, map_location='cpu', weights_only=False)
        ex.validate_resume_recipe(state)
        vocabulary = state['vocabulary']
        lookup = {word: i + 1 for i, word in enumerate(vocabulary)}
        model = MLTSF(len(vocabulary) + 1).cuda()
        model.load_state_dict(state['model'])
        assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
        report['checkpoints'][label] = {'sha256': checksum, 'epoch': state['epoch']}
        modes = ['running', 'video_all', 'video_spatial', 'video_temporal'] if label == 'best52' else ['running', 'video_all']
        report['results'][label] = {}
        for split, rs in selected.items():
            paths = {row['id']: ex.frame_paths(row, split) for row in rs}
            predictions = {mode: {} for mode in modes}
            observations = {mode: [] for mode in modes}
            baseline_matches = 0
            with ex.VideoLoader() as loader:
                for number, (row, (video, _)) in enumerate(loader.iterate(rs, paths), 1):
                    if time.monotonic() - start > wall_seconds - 100:
                        raise TimeoutError('Diagnostic time limit; retain partial artifacts')
                    video = video.cuda()
                    target_words = row['annotation'].split()
                    oov = [w for w in target_words if w not in lookup]
                    target = None if oov else torch.tensor([lookup[w] for w in target_words], device='cuda')
                    for mode in modes:
                        set_normalization(model, mode)
                        logits, _ = model(video)
                        logp = logits.log_softmax(-1)
                        loss = None if target is None else F.ctc_loss(logp[:, None], target, [len(logits)], [len(target)], reduction='sum')
                        assert loss is None or torch.isfinite(loss)
                        path_indices = logits.argmax(-1)
                        words = [vocabulary[i - 1] for i in collapse(path_indices.tolist())]
                        predictions[mode][row['id']] = words
                        observations[mode].append({'id': row['id'], 'ctc_nll': None if loss is None else float(loss),
                            'oov_tokens': oov, 'target_tokens': len(target_words), 'predicted_tokens': len(words),
                            'blank_argmax_fraction': float((path_indices == 0).float().mean()),
                            'mean_blank_probability': float(logp[:, 0].exp().mean())})
                        if label == 'best52' and split == 'dev' and mode == 'running':
                            baseline_matches += words == original_dev[row['id']]
                    if number % 16 == 0:
                        ex.write_json(out / 'progress.json', {'checkpoint': label, 'split': split,
                                      'samples_done': number, 'elapsed_seconds': time.monotonic() - start})
                        print(json.dumps({'checkpoint': label, 'split': split, 'samples_done': number}), flush=True)
            results = {}
            for mode in modes:
                dest = out / label / split / mode
                metrics = ex.score(predictions[mode], split, dest)
                obs = observations[mode]
                valid_nll = [x for x in obs if x['ctc_nll'] is not None]
                ex.write_json(dest / 'per-video.json', obs)
                results[mode] = {'metrics': metrics,
                    'ctc_nll_samples': len(valid_nll),
                    'oov_samples': len(obs) - len(valid_nll),
                    'oov_token_count': sum(len(x['oov_tokens']) for x in obs),
                    'mean_ctc_nll': sum(x['ctc_nll'] for x in valid_nll) / len(valid_nll),
                    'ctc_nll_per_reference_token': sum(x['ctc_nll'] for x in valid_nll) / sum(x['target_tokens'] for x in valid_nll),
                    'mean_blank_argmax_fraction': sum(x['blank_argmax_fraction'] for x in obs) / len(obs),
                    'predicted_to_target_token_ratio': sum(x['predicted_tokens'] for x in obs) / sum(x['target_tokens'] for x in obs)}
            if label == 'best52' and split == 'dev':
                report['baseline_dev_predictions_matching_original'] = baseline_matches
                assert baseline_matches == len(rs), 'Diagnostic running-stat baseline differs from original predictions'
            report['results'][label][split] = results
            ex.write_json(out / 'diagnostic.json', report)
        assert all(torch.equal(v.cpu(), state['model'][k]) for k, v in model.state_dict().items()), 'Weights or buffers changed'
        assert ex.sha(path) == checksum
        report['checkpoints'][label]['parameters_and_buffers_unchanged'] = True
        del state, model
        torch.cuda.empty_cache()
    report.update(finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - start, completed=True)
    ex.write_json(out / 'diagnostic.json', report)
    print(json.dumps({'completed': True, 'results': report['results']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['bn_diagnostic'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=1500)
    args = parser.parse_args()
    run(args.out, args.wall_seconds)
