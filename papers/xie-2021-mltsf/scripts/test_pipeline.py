"""Pixel/RNG equivalence and interruption tests for the actual training driver."""
from contextlib import closing
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import experiment as ex


class TinyModel(torch.nn.Module):
    def __init__(self, classes, gfe=False, checkpoint_backbone=False):
        super().__init__()
        self.layer = torch.nn.Linear(3, classes)
        self.aux = torch.nn.Linear(3, classes) if gfe else None

    def forward(self, video):
        x = video[::4].mean((2, 3))
        return self.layer(x), self.aux(x) if self.aux is not None else None


class Pipeline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = []
        for i in range(16):
            path = self.root / f'{i}.png'
            pixels = np.random.default_rng(i).integers(0, 256, (30, 40, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            self.paths.append(str(path))
        self.rs = [{'id': str(i), 'annotation': 'A B'} for i in range(3)]
        self.inventory = {r['id']: self.paths for r in self.rs}

    def tearDown(self):
        ex.STOP_REQUESTED = False
        self.temp.cleanup()

    def test_pixels_and_rng_match_original_loader(self):
        # Independent copy of the original preprocessing, including duplicate decodes.
        random.seed(9)
        expected = []
        for _ in self.rs:
            n = len(self.paths)
            size = max(16, round(n * random.uniform(.8, 1.2)))
            indices = np.rint(np.linspace(0, n-1, size)).astype(np.int64)
            left, top = random.randrange(33), random.randrange(33)
            frames = []
            for i in indices:
                with Image.open(self.paths[i]) as im:
                    im = im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
                    frames.append(np.array(im.crop((left, top, left+224, top+224))))
            expected.append((torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()/255,
                             torch.from_numpy(indices), random.getstate()))
        random.seed(9)
        with ex.VideoLoader() as loader:
            for (_, (x, indices)), (want, want_indices, rng) in zip(
                    loader.iterate(self.rs, self.inventory, True), expected):
                self.assertTrue(torch.equal(x, want))
                self.assertTrue(torch.equal(indices, want_indices))
                self.assertEqual(random.getstate(), rng)

    def test_discarded_prefetch_does_not_advance_rng(self):
        random.seed(12)
        with ex.VideoLoader() as loader, closing(loader.iterate(self.rs, self.inventory, True)) as it:
            next(it)
            state = random.getstate()
        self.assertEqual(random.getstate(), state)
        expected = ex.load_video(self.paths, augment=True)
        random.setstate(state)
        with ex.VideoLoader() as loader, closing(loader.iterate(self.rs[1:], self.inventory, True)) as it:
            _, actual = next(it)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_ctc_guard_rejects_only_infeasible_temporal_draws(self):
        row = {'annotation': ' '.join(str(i) for i in range(16))}
        minimum = ex.minimum_ctc_frames(row)
        self.assertEqual(minimum, 64)
        expected_rng = random.Random(1)
        rejected = round(74*expected_rng.uniform(.8, 1.2))
        self.assertLess(rejected, minimum)
        accepted = round(74*expected_rng.uniform(.8, 1.2))
        crop = (expected_rng.randrange(33), expected_rng.randrange(33))
        actual_rng = random.Random(1)
        indices, left, top = ex.video_plan(74, True, actual_rng, minimum)
        self.assertEqual(len(indices), accepted)
        self.assertEqual((left, top), crop)
        self.assertEqual(actual_rng.getstate(), expected_rng.getstate())
        self.assertEqual(ex.minimum_ctc_frames({'annotation': 'A A B'}), 16)
        with self.assertRaises(ValueError):
            ex.video_plan(16, True, random.Random(1), 68)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_resume_matches_uninterrupted_training_and_alignment(self):
        original_step, original_align = ex.train_step, ex.forced_alignment
        for interrupt_phase in ('train', 'alignment', 'dev'):
            with self.subTest(interrupt_phase=interrupt_phase):
                full, resumed = self.root / f'full-{interrupt_phase}', self.root / f'resumed-{interrupt_phase}'
                full.mkdir()
                resumed.mkdir()
                # The real driver reaches the GFE refresh at epoch 16 on tiny data.
                def evaluate(*args):
                    return {'wer': 50.}
                with patch.object(ex, 'MLTSF', TinyModel), patch.object(ex, 'rows', return_value=self.rs), \
                     patch.object(ex, 'frame_paths', return_value=self.paths), patch.object(ex, 'evaluate', side_effect=evaluate):
                    ex.train(full, True, False, 10000, device='cpu', epochs=16)
                    count = 0
                    def step(*args, **kwargs):
                        nonlocal count
                        result = original_step(*args, **kwargs)
                        count += 1
                        if interrupt_phase == 'train' and count == 2:
                            ex.request_stop(None, None)
                        if interrupt_phase == 'dev' and count == len(self.rs):
                            ex.request_stop(None, None)
                        return result
                    def align(*args, **kwargs):
                        result = original_align(*args, **kwargs)
                        if interrupt_phase == 'alignment':
                            ex.request_stop(None, None)
                        return result
                    with patch.object(ex, 'train_step', side_effect=step), patch.object(ex, 'forced_alignment', side_effect=align):
                        ex.train(resumed, True, False, 10000, device='cpu', epochs=16)
                    self.assertTrue((resumed / 'pause.json').exists())
                    state = torch.load(resumed / 'last.pt', weights_only=False)
                    self.assertEqual(state['progress']['phase'], interrupt_phase)
                    ex.STOP_REQUESTED = False
                    ex.train(resumed, True, False, 10000, device='cpu', epochs=16)
                self.assert_nested_equal(torch.load(full / 'last.pt', weights_only=False),
                                         torch.load(resumed / 'last.pt', weights_only=False))
                self.assert_nested_equal(torch.load(full / 'best.pt', weights_only=False),
                                         torch.load(resumed / 'best.pt', weights_only=False))
                self.assertFalse((resumed / 'pause.json').exists())

    def test_approved_checkpoint_migration_is_narrow(self):
        import copy
        order = [f'row-{i}' for i in range(5672)]
        order[4686] = ex.BROKEN_TRAIN_ID
        state = {'recipe_sha256': dict(ex.PRE_EXCLUSION_RECIPE), 'epoch': 0,
                 'progress': {'phase': 'train', 'next_index': 4600, 'order': order},
                 'alignments': {}, 'model': {'weight': torch.tensor([1., 2.])},
                 'optimizer': {'state': 'preserve'}, 'python_rng': random.getstate()}
        migrated = copy.deepcopy(state)
        ex.validate_resume_recipe(migrated)
        self.assertEqual(len(migrated['progress']['order']), 5671)
        self.assertEqual(migrated['progress']['next_index'], 4600)
        self.assertEqual(migrated['progress']['order'][:4600], order[:4600])
        for key in ('model', 'optimizer', 'python_rng'):
            self.assert_nested_equal(migrated[key], state[key])
        for alteration in ('unknown_recipe', 'already_processed', 'later_epoch'):
            rejected = copy.deepcopy(state)
            if alteration == 'unknown_recipe':
                rejected['recipe_sha256']['experiment.py'] = 'unknown'
            elif alteration == 'already_processed':
                rejected['progress']['next_index'] = 4700
            else:
                rejected['epoch'] = 1
            with self.assertRaises(AssertionError):
                ex.validate_resume_recipe(rejected)


if __name__ == '__main__':
    unittest.main()
