"""Tests for scientific invariants, independent of model-score closeness."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from search import decode, initialize, propose, leaders_from_history


class FixedRandom:
    def uniform(self, *args, size=None):
        return np.full(size, 0.5) if size is not None else 0.5

    def normal(self, size):
        return np.zeros(size)

    def standard_cauchy(self, size):
        return np.zeros(size)


class ProtocolTests(unittest.TestCase):
    def test_log_endpoints_and_initial_base(self):
        self.assertEqual(decode([0]), {"head_lr": .001, "backbone_multiplier": 1.0})
        self.assertEqual(decode([1]), {"head_lr": .01, "backbone_multiplier": .1})
        a = initialize(np.random.default_rng(42))
        np.testing.assert_array_equal(a[0], [1])
        np.testing.assert_array_equal(a, initialize(np.random.default_rng(42)))

    def test_literal_equations_and_no_intermediate_evaluations(self):
        leaders = np.array([[.2, .3], [.4, .5], [.6, .7]])
        positions, trace = propose(np.array([[.1, .1]]), leaders, np.array([.1, .2, .3]), 0, FixedRandom(), {})
        # r1=.5 makes A=0, so proposals equal leaders. Noise=0 leaves
        # their average .4,.5; crow Eq.13 moves halfway toward .3,.4.
        np.testing.assert_allclose(positions[0], [.35, .45])
        np.testing.assert_allclose(trace[0]["weighted"], [(.2+2*.4+3*.6)/18, (.3+2*.5+3*.7)/18])
        self.assertFalse(trace[0]["equation12_repaired_without_evaluation"])

    def test_zero_fitness_weights_and_cached_tie(self):
        leaders = np.full((3, 2), .6)
        cache = {(.6, .6): .1, (.19999999999999998, .19999999999999998): .1}
        _, trace = propose(np.full((1, 2), .2), leaders, np.zeros(3), 0, FixedRandom(), {})
        np.testing.assert_allclose(trace[0]["weighted"], [.2, .2])
        cache = {tuple(trace[0]["mutated"]): .1, tuple(trace[0]["weighted"]): .1}
        result, trace = propose(np.full((1, 2), .2), leaders, np.zeros(3), 0, FixedRandom(), cache)
        self.assertTrue(trace[0]["equation12_repaired_without_evaluation"])
        np.testing.assert_allclose(result[0], [.4, .4])  # tie retains weighted .2, crow halfway to .6

    def test_earlier_evaluation_wins_tie(self):
        history = [{"evaluation": i, "position": [i/10, i/10], "misclassified": 100} for i in [3, 1, 2, 0]]
        leaders, _ = leaders_from_history(history)
        np.testing.assert_allclose(leaders[:, 0], [0, .1, .2])

    def test_accuracy_is_not_multiclass_accuracy(self):
        from experiment import metric_report
        cm = np.eye(29, dtype=int) * 10
        cm[0, 0] -= 2
        cm[0, 1] += 2
        result = metric_report(cm)
        self.assertEqual(result["misclassified"], 2)
        self.assertAlmostEqual(result["multiclass_accuracy_percent"], 100*288/290)
        self.assertAlmostEqual(result["macro_percent"]["accuracy"], 100*(1-4/(29*290)))
        self.assertEqual(result["zero_denominators"], [])

    def test_frozen_cnns_keep_parameters_and_buffers_fixed(self):
        import torch
        from torch import nn
        from experiment import Recognizer
        model = Recognizer.__new__(Recognizer)
        nn.Module.__init__(model)
        model.resnext = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        model.vgg = nn.Linear(3, 3)
        model.vit = nn.Linear(3, 3)
        model.head = nn.Linear(3, 2)
        before = {k: v.clone() for k, v in model.resnext.state_dict().items()}
        model.stage(True)
        out = model.head(model.vit(model.vgg(model.resnext(torch.randn(5, 3)))))
        out.sum().backward()
        self.assertTrue(all(p.grad is None for m in (model.resnext, model.vgg) for p in m.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.vit.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.resnext.state_dict().items()))
        self.assertFalse(model.resnext.training)
        model.stage(False)
        self.assertFalse(model.vit.training)
        self.assertTrue(all(not p.requires_grad for p in model.vit.parameters()))
        self.assertTrue(model.head.training)

    def test_cached_campaign_keeps_search_budget_and_final_epochs(self):
        from experiment import campaign
        calls = []
        def candidate(output, cache_path, params, epochs):
            calls.append(("search", epochs))
            return {"misclassified": 100}
        def final(output, params, epochs, final=False, until_epoch=None):
            calls.append(("final", until_epoch))
            self.assertAlmostEqual(params["head_lr"] * params["backbone_multiplier"], .001)
            return {}
        with tempfile.TemporaryDirectory() as temp, patch("cached_search.train_candidate", side_effect=candidate), patch("experiment.train_configuration", side_effect=final):
            root = Path(temp)
            (root / "feature-cache").mkdir()
            (root / "feature-cache/metadata.json").write_text("{}")
            for _ in range(30):
                result = campaign(root)
            self.assertTrue(result["complete"])
            self.assertEqual(calls[:20], [("search", 10)] * 20)
            self.assertEqual(calls[20:], [("final", i) for i in range(5, 51, 5)])

    def test_feature_cache_reads_rows_and_labels(self):
        from cached_search import Features
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.npy"
            x = np.arange(2*49*3328, dtype=np.float32).reshape(2, 49, 3328)
            np.save(path, x)
            ds = Features(path, [{"label": 7}, {"label": 3}])
            value, label = ds[1]
            np.testing.assert_array_equal(value.numpy(), x[1])
            self.assertEqual(label, 3)


if __name__ == "__main__":
    unittest.main()
