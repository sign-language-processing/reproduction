"""Core algorithm checks; these are not a real-data reproduction preflight."""
import itertools
import unittest
import torch
from model import MLTSF, collapse, forced_alignment, mapped_alignment, select_neighbors


class Algorithms(unittest.TestCase):
    def test_viterbi_against_exhaustive_paths(self):
        torch.manual_seed(1)
        probabilities = torch.randn(5, 3).log_softmax(-1)
        for target in ([1, 2], [1, 1], [2], []):
            valid = [p for p in itertools.product(range(3), repeat=5) if collapse(p) == list(target)]
            expected = max(sum(probabilities[t, p[t]].item() for t in range(5)) for p in valid)
            actual = forced_alignment(probabilities, target)
            self.assertEqual(collapse(actual.tolist()), list(target))
            self.assertAlmostEqual(sum(probabilities[t, actual[t]].item() for t in range(5)), expected, places=6)
        with self.assertRaises(ValueError):
            forced_alignment(torch.randn(2, 3).log_softmax(-1), [1, 1])

    def test_neighbors_and_time_mapping(self):
        x = torch.randn(40, 12)
        selected = select_neighbors(x, 8)
        self.assertTrue(torch.all((selected - torch.arange(40)[:, None]).abs() <= 8))
        self.assertTrue(torch.all(selected[:, 1:] > selected[:, :-1]))
        canonical = torch.tensor([1, 0, 2, 0])
        self.assertTrue(torch.equal(mapped_alignment(canonical, torch.arange(16), 4), canonical))
        self.assertEqual(mapped_alignment(canonical, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), 2).tolist(), [1, 1])

    def test_checkpoint_preserves_gradients_and_bn(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        plain = MLTSF(4, gfe=True).double()
        checkpointed = MLTSF(4, gfe=True, checkpoint_backbone=True).double()
        checkpointed.load_state_dict(plain.state_dict())
        x = torch.randn(16, 3, 32, 32, dtype=torch.double)
        for model in (plain, checkpointed):
            logits, auxiliary = model(x)
            self.assertEqual(tuple(logits.shape), (4, 4))
            (logits.square().sum()+auxiliary.square().sum()).backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, msg=name)
        for (name, p), (_, q) in zip(plain.named_buffers(), checkpointed.named_buffers()):
            torch.testing.assert_close(p, q, msg=name)


if __name__ == '__main__':
    unittest.main()
