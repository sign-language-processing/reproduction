"""Independent implementation of arXiv:2107.12762v1 Tables 1–2.

All experimental inferences are documented in reproduction.json. Input is one
complete RGB video, T x 3 x H x W; blank is vocabulary index zero.
"""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def select_neighbors(features, k):
    """Equations 5–7: mask +/- k, top-k, then restore chronological order."""
    t, c = features.shape
    if t < k:
        raise ValueError(f"{t} frames cannot supply {k} distinct neighbors")
    similarity = (features @ features.T / c).softmax(-1)
    positions = torch.arange(t, device=features.device)
    similarity = similarity.masked_fill((positions[:, None] - positions).abs() > k, -torch.inf)
    return similarity.topk(k, dim=-1).indices.sort(dim=-1).values


class PTC(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.position = nn.Embedding(2 * k + 1, 512)
        self.convs = nn.ModuleList([nn.Conv1d(512, 512, 3, padding=1) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(512) for _ in range(2)])
        self.mlp = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512))

    def forward(self, x):
        indices = select_neighbors(x, self.k)
        relative = indices - torch.arange(len(x), device=x.device)[:, None]
        y = x[indices] + self.position(relative + self.k)
        for conv, norm in zip(self.convs, self.norms):
            y = F.relu(norm(conv(y.transpose(1, 2)).transpose(1, 2)))
        return self.mlp(y.amax(dim=1))


@contextlib.contextmanager
def preserve_bn_buffers(module):
    """Checkpoint recomputation must not update running statistics a second time."""
    buffers = [(b, b.clone()) for name, b in module.named_buffers()
               if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)


class MLTSF(nn.Module):
    def __init__(self, classes, gfe=False, checkpoint_backbone=False):
        super().__init__()
        layers = []
        cin = 3
        # Table 2 puts pooling after the FIRST convolution of each channel pair.
        for i, cout in enumerate([32, 32, 64, 64, 128, 128, 256, 256, 512, 512]):
            layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
            if i % 2 == 0:
                layers.append(nn.MaxPool2d(2))
            cin = cout
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(1)]
        self.backbone = nn.Sequential(*layers)
        self.scales = nn.ModuleList([PTC(k) for k in (16, 12, 8)])
        self.aggregate = nn.Linear(512, 3)
        self.level1 = nn.Sequential(nn.Conv1d(512, 512, 5, padding=2), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.MaxPool1d(2), nn.Conv1d(512, 512, 5, padding=2),
                                    nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2))
        self.level2 = nn.Sequential(nn.Conv1d(512, 1024, 3, padding=1), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(1024, classes)
        self.gfe = nn.Linear(512, classes) if gfe else None
        self.checkpoint_backbone = checkpoint_backbone

    def forward(self, video):
        if self.training and self.checkpoint_backbone:
            x = checkpoint(self.backbone, video, use_reentrant=False,
                           context_fn=lambda: (contextlib.nullcontext(), preserve_bn_buffers(self.backbone)))
        else:
            x = self.backbone(video)
        weights = self.aggregate(x).softmax(-1)
        x = (torch.stack([scale(x) for scale in self.scales], dim=1) * weights[:, :, None]).sum(1)
        gloss = self.level1(x.T.unsqueeze(0))
        logits = self.classifier(self.level2(gloss).squeeze(0).T)
        auxiliary = self.gfe(gloss.squeeze(0).T) if self.gfe is not None else None
        return logits, auxiliary


def forced_alignment(log_probs, target):
    """Ground-truth-constrained CTC Viterbi path (blank=0); deterministic ties."""
    p = log_probs.detach().cpu().double()
    labels = torch.zeros(2 * len(target) + 1, dtype=torch.long)
    labels[1::2] = torch.as_tensor(target, dtype=torch.long).cpu()
    t, _ = p.shape
    n = len(labels)
    prev = torch.full((n,), -torch.inf, dtype=torch.double)
    prev[0] = p[0, 0]
    if n > 1:
        prev[1] = p[0, labels[1]]
    trace = torch.zeros(t, n, dtype=torch.long)
    can_skip = torch.zeros(n, dtype=torch.bool)
    can_skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    for step in range(1, t):
        one = F.pad(prev[:-1], (1, 0), value=-torch.inf)
        two = F.pad(prev[:-2], (2, 0), value=-torch.inf) if n > 1 else torch.full_like(prev, -torch.inf)
        candidates = torch.stack([prev, one, two.masked_fill(~can_skip, -torch.inf)])
        values, moves = candidates.max(0)
        prev = values + p[step, labels]
        trace[step] = moves
    end = n - 1 if n == 1 or prev[-1] >= prev[-2] else n - 2
    if not torch.isfinite(prev[end]):
        raise ValueError('No valid CTC path; target cannot fit available encoder steps')
    path = [end]
    for step in range(t - 1, 0, -1):
        end -= int(trace[step, end])
        path.append(end)
    return labels[torch.tensor(path[::-1])]


def mapped_alignment(canonical, frame_indices, output_length):
    indices = torch.arange(output_length) * 4
    indices = frame_indices.cpu()[indices.clamp_max(len(frame_indices) - 1)] // 4
    return canonical[indices.clamp_max(len(canonical) - 1)]


def gfe_loss(logits, labels):
    weights = torch.where(labels == 0, (labels != 0).float().mean(), 1.)
    return (F.cross_entropy(logits, labels, reduction='none') * weights).mean()


def collapse(path):
    return [int(x) for i, x in enumerate(path) if x != 0 and (i == 0 or path[i - 1] != x)]
