"""Bounded real-data cost benchmark; never produces a Table 3 result."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import random
import resource
import subprocess
import time

import numpy as np
import torch
import experiment as ex


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    for test in ('test_model.py', 'test_pipeline.py'):
        subprocess.run(['python', str(Path(__file__).with_name(test))], check=True)
    ex.seed_all()
    rs = ex.rows('train')
    vocabulary = sorted({w for r in rs for w in r['annotation'].split()})
    with ThreadPoolExecutor(max_workers=32) as pool:
        inventories = list(pool.map(lambda row: ex.frame_paths(row, 'train'), rs))
    paths = {r['id']: files for r, files in zip(rs, inventories)}
    ordered = sorted(rs, key=lambda row: len(paths[row['id']]))
    # Adjacent-length pairs at eight quantiles; alternate first-read method order.
    pairs = [[ordered[int(q*(len(rs)-2))+j] for j in range(2)] for q in np.linspace(.05, .95, 8)]
    selected = [row for pair in pairs for row in pair]
    result = {'first_reads': [], 'warm_training': {}, 'conditional': True,
              'first_read_caveat': 'First read in this process, not a controlled eviction of platform caches.',
              'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': subprocess.getoutput('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader')},
              'source_sha256': {p.name: ex.sha(p) for p in Path(__file__).parent.glob('*.py')}}
    def record():
        ex.write_json(out / 'benchmark.json', result)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, pair in enumerate(pairs):
            for j, row in enumerate(pair):
                parallel = (i+j) % 2 == 1
                start = time.monotonic()
                x, _ = ex.load_video(paths[row['id']], pool=pool if parallel else None)
                seconds = time.monotonic()-start
                entry = {'frames': len(x), 'parallel': parallel, 'seconds': seconds}
                result['first_reads'].append(entry)
                print(json.dumps({'first_read': entry}), flush=True)
                if i == 0:
                    other, _ = ex.load_video(paths[row['id']], pool=None if parallel else pool)
                    assert torch.equal(x, other), 'Parallel decoding changed pixels'
                    del other
                del x
                record()
    model = ex.MLTSF(len(vocabulary)+1, gfe=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ex.save_checkpoint(out / 'initial.pt', model, optimizer, 0, float('inf'), vocabulary, {})
    # Validate all actual augmented inputs independently of nondeterministic CUDA
    # training, outside the timing window. Keep hashes, not dataset pixels.
    inputs = {}
    for method in ('serial', 'prefetch'):
        ex.restore(out / 'initial.pt', model, optimizer)
        digests = []
        with ex.VideoLoader() as loader:
            iterator = (loader.iterate(selected, paths, True) if method == 'prefetch' else
                        ((r, ex.load_video(paths[r['id']], True, min_frames=ex.minimum_ctc_frames(r))) for r in selected))
            for row, (x, indices) in iterator:
                h = hashlib.sha256(x.numpy().tobytes())
                h.update(indices.numpy().tobytes())
                digests.append(h.hexdigest())
        inputs[method] = digests
        inputs[method+'_rng'] = random.getstate()
    assert inputs['serial'] == inputs['prefetch']
    assert inputs['serial_rng'] == inputs['prefetch_rng']
    result['real_input_equivalence'] = {'passed': True, 'sha256': inputs['serial']}
    # An unreported warm-up and repeated serial control separate initial kernel
    # setup and GPU nondeterminism from prefetch overhead.
    for method in ('serial_warmup', 'serial', 'serial_repeat', 'prefetch'):
        ex.restore(out / 'initial.pt', model, optimizer)
        torch.cuda.synchronize()
        started = time.monotonic()
        entries = []
        with ex.VideoLoader() as loader:
            iterator = (loader.iterate(selected, paths, True) if method == 'prefetch' else
                        ((r, ex.load_video(paths[r['id']], True, min_frames=ex.minimum_ctc_frames(r))) for r in selected))
            for row, prepared in iterator:
                torch.cuda.synchronize()
                begin = time.monotonic()
                loss, frames = ex.train_step(model, optimizer, row, vocabulary, paths[row['id']], prepared=prepared)
                torch.cuda.synchronize()
                entries.append({'frames': frames, 'loss': loss, 'compute_seconds': time.monotonic()-begin})
        result['warm_training'][method] = {'seconds': time.monotonic()-started, 'steps': entries}
        record()
    reference = result['warm_training']['serial']['steps']
    result['cuda_repeat_variation'] = {}
    for method in ('serial_repeat', 'prefetch'):
        actual = result['warm_training'][method]['steps']
        assert [x['frames'] for x in reference] == [x['frames'] for x in actual]
        assert abs(reference[0]['loss']-actual[0]['loss']) < .001
        result['cuda_repeat_variation'][method] = [b['loss']-a['loss'] for a, b in zip(reference, actual)]
    result['cuda_repeat_note'] = ('CUDA CTC gradients are nondeterministic; tiny differences can change discrete '
                                 'top-k neighbors and grow over updates. Compare serial-repeat control; '
                                 'require exact pixels/RNG and checkpoint restoration, not multi-step GPU bitwise equality.')
    record()
    longest = ordered[-1]
    files = paths[longest['id']]
    indices = np.rint(np.linspace(0, len(files)-1, round(len(files)*1.2))).astype(np.int64)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=8) as pool:
        canonical, _ = ex.load_video(files, pool=pool)
        model.eval()
        with torch.no_grad():
            logits, _ = model(canonical.cuda())
            labels = [vocabulary.index(w)+1 for w in longest['annotation'].split()]
            start = time.monotonic()
            alignment = ex.forced_alignment(logits.log_softmax(-1), labels)
            result['alignment_only_seconds'] = time.monotonic()-start
        del canonical, logits
        prepared = ex.load_video(files, plan=(indices, 0, 32), pool=pool)
    loss, frames = ex.train_step(model, optimizer, longest, vocabulary, files, alignment, prepared=prepared)
    torch.cuda.synchronize()
    result['maximum_length_step'] = {'frames': frames, 'loss': loss,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory}
    del prepared
    ex.save_checkpoint(out / 'roundtrip.pt', model, optimizer, 0, float('inf'), vocabulary, {},
                       {'phase': 'train', 'next_index': 1, 'order': [r['id'] for r in selected]})
    # Verify resumed next optimizer update against the uninterrupted update on real data.
    row = selected[0]
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    loss_a, _ = ex.train_step(model, optimizer, row, vocabulary, paths[row['id']])
    state_a = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng_a = random.getstate()
    restored = ex.restore(out / 'roundtrip.pt', model, optimizer)
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), before[k]), k
    for k, state in optimizer.state_dict()['state'].items():
        for key, value in state.items():
            expected = restored['optimizer']['state'][k][key]
            assert torch.equal(value.cpu(), expected.cpu()) if isinstance(value, torch.Tensor) else value == expected
    loss_b, _ = ex.train_step(model, optimizer, row, vocabulary, paths[row['id']])
    max_delta = max((v.cpu().double()-state_a[k].double()).abs().max().item() for k, v in model.state_dict().items())
    assert abs(loss_a-loss_b) < .001
    assert random.getstate() == rng_a
    result['real_resume_check'] = {'uninterrupted_loss': loss_a, 'resumed_loss': loss_b, 'passed': True,
                                   'restored_state_exact': True, 'post_update_max_parameter_or_buffer_delta': max_delta}
    result['tiny_dev'] = ex.evaluate(model, ex.rows('dev')[:3], 'dev', vocabulary, out / 'dev-tiny')
    result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for name in ('memory.peak', 'memory.current'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result['cgroup_'+name.replace('.', '_')+'_bytes'] = int(path.read_text())
    result['passed'] = True
    record()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['benchmark'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int)
    main(parser.parse_args().out)
