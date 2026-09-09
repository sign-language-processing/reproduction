"""OLMD reimplementation from the paper (preference level 3).

Reference: Yu et al., "OLMD: Orientation-aware Long-term Motion Decoupling for
Continuous Sign Language Recognition", AAAI-25. arXiv:2503.08205v1.

No code was ever published for this paper, and the MP-Block that the paper
defers to "supplementary materials" does not exist in any public form. Every
inferred choice below carries an INFERRED tag naming the evidence that
constrains it. See reproduction.json.guesses for the full ledger; the guess_id
in each tag is the key there.

Equation numbers refer to the arXiv v1 text.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

# Channel widths of ResNet34 stages 2, 3 and 4, where the paper attaches its
# modules (Table 3 selects exactly these three locations).
STAGE_CHANNELS = (128, 256, 512)

# INFERRED (guess_id: lma-reduction-ratio). The paper writes "the channel
# dimension is compressed to C/r by a CNN" and never gives r. Table 5's
# parameter column rises +0.22M per +2 context, and Eq. 10's concatenation
# convolution is the only k-dependent weight, contributing 2*(C/r)^2 summed
# over the three stages. That predicts +0.688M (r=1), +0.172M (r=2), +0.043M
# (r=4), +0.0027M (r=16). The observed +0.220M selects r=2 and excludes the
# lineage's house r=16 by roughly 80x.
LMA_REDUCTION = 2


class LMA(nn.Module):
    """Long-term Motion Aggregation, Eqs. 7-11.

    Fully specified by the paper apart from the reduction ratio, the ConvBlock's
    internal widths, and the sequence-boundary rule.
    """

    def __init__(self, channels: int, context: int = 9, reduction: int = LMA_REDUCTION):
        super().__init__()
        if context % 2 != 1:
            raise ValueError(f"context length must be odd, got {context}")
        self.context = context
        self.half = context // 2
        hidden = max(1, channels // reduction)
        self.hidden = hidden

        # "the channel dimension is compressed to C/r by a CNN" (Eq. 7 preamble).
        # INFERRED (guess_id: unstated-training-hyperparameters): the paper does
        # not say whether this compression carries norm/activation. Kept bare so
        # the residual in Eq. 11 stays the only nonlinearity boundary.
        self.compress = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)

        # Eq. 9: a shared ConvBlock applied to each difference map, "mainly
        # consists of three CNNs, each with a kernel size of 1x3x3" -- i.e. 3D
        # convs with no temporal extent, so S_(2n+1) is the only temporal mixing.
        # INFERRED (guess_id: mp-block-internals, same family of unstated
        # internals): widths held at `hidden`, BatchNorm + ReLU between convs.
        block = []
        for index in range(3):
            block += [
                nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(hidden),
            ]
            if index < 2:
                block.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*block)

        # Eq. 10: Conv_{1x1x1} over the concatenated difference maps, back to C/r.
        # INFERRED (guess_id: eq10-concat-inconsistency): Eq. 10 writes the list
        # as [C_(-n),...,C_(n)] (2n+1 maps, including the identically-zero centre
        # difference), but Eq. 9 defines differences only for i != 0 and Fig. 4
        # labels the concatenated tensor "4C" at context 5, which is 2n = 4 maps.
        # Fig. 4 and Eq. 9 are followed: 2n non-zero maps.
        self.fuse = nn.Conv3d(hidden * 2 * self.half, hidden, kernel_size=1, bias=False)

        # Eq. 11: restore to C, then residual.
        self.expand = nn.Conv3d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T, H, W) -> same shape."""
        compressed = self.compress(x)

        # Eq. 7: sliding window of size 2n+1 centred on each frame.
        # INFERRED (guess_id: batching-and-padding): the paper does not state
        # the sequence-boundary rule, which affects `context - 1` frames of
        # every clip. Replicate padding is used, matching how the lineage pads
        # video (frame replication, never zeros), so boundary differences are
        # zero rather than spuriously large.
        padded = F.pad(compressed, (0, 0, 0, 0, self.half, self.half), mode="replicate")

        centre = compressed
        diffs = []
        for offset in range(-self.half, self.half + 1):
            if offset == 0:
                continue  # Eq. 8 defines differences for i != 0
            start = offset + self.half
            neighbour = padded[:, :, start : start + compressed.shape[2]]
            # Eq. 8: C'_(i) = C_(0) - C_(i), the centre frame minus the neighbour.
            diffs.append(self.conv_block(centre - neighbour))

        fused = self.fuse(torch.cat(diffs, dim=1))
        return self.expand(fused) + x  # Eq. 11


class MPBlock(nn.Module):
    """The motion-purification mixer of Eq. 2.

    BLOCKING INFERENCE (guess_id: mp-block-internals). The paper says only
    "The MP-Block (2D CNNs, detailed in the supplementary materials)". That
    supplementary material does not exist: arXiv carries only v1 with no
    ancillary files, the 9-page PDF has no appendix or embedded files, and the
    AAAI proceedings entry exposes a PDF and poster only.

    Constraints that do hold:
      * Eqs. 2-3 form the MetaFormer/PoolFormer macro-block -- a token mixer
        with its own residual, then an FFN -- not ConvNeXt, which has a single
        residual and a mid-block norm.
      * The operand is (C, T, W) or (C, T, H): a channels-by-time-by-one-spatial
        -axis plane treated as a 2D image.
      * C_out = C_in = C is forced, because Eq. 5 adds US(X_h) back into a
        C-channel trunk.
      * The plane is only 7 wide at stage 4, so a square 7x7 kernel is
        geometrically implausible. The lineage uses anisotropic kernels with a
        long temporal extent (CorrNet's mixers are (9,3,3) depthwise), so a
        (9,3) kernel over (T, spatial) follows that convention.

    Depth, dense-vs-depthwise, and the exact kernel remain free choices.
    """

    def __init__(self, channels: int, temporal_kernel: int = 9, spatial_kernel: int = 3):
        super().__init__()
        padding = (temporal_kernel // 2, spatial_kernel // 2)
        # INFERRED: depthwise, mirroring the lineage's depthwise mixers, which
        # also keeps the parameter cost within the Table 5 budget.
        self.mixer = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(temporal_kernel, spatial_kernel),
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class FFN(nn.Module):
    """Eq. 3: FFN(X) = max(0, Norm(X W_1)) W_2.

    W_1 and W_2 are 1x1 2D convolutions and Norm is layer normalization, both
    stated. INFERRED (guess_id: ffn-and-ds-widths): the expansion ratio is not
    given; 4 is the universal convention (Vaswani, PoolFormer, ConvNeXt, ViT,
    Swin). C_out = C_in is forced by Eq. 5.

    Note the norm placement: Eq. 3 puts normalization *between* W_1 and W_2,
    which matches none of the reference architectures (all normalize before the
    FFN). The equation is followed literally rather than silently "corrected";
    recorded as guess_id: mp-block-internals.
    """

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = channels * ratio
        self.w1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, hidden)  # LayerNorm over (C,H,W) for conv maps
        self.w2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.norm(self.w1(x)), inplace=True))


class OMP(nn.Module):
    """Orientation-aware Motion Purification for one stage, Eqs. 1-4.

    Decoupling (Eq. 1) pools one spatial axis away: X_h keeps W, X_v keeps H.
    Table 7 selects AvgPool. HMP acts on X_h and VMP on X_v, per Eq. 2, Fig. 3
    and the Fig. 2 caption. (The OMP paragraph's clause "for purifying vertical
    and horizontal motions, respectively" contradicts all three and is treated
    as a writing error -- guess_id: hmp-vmp-naming-inconsistency.)
    """

    def __init__(self, channels: int, prev_channels: int | None = None, ffn_ratio: int = 4):
        super().__init__()
        self.hmp = MPBlock(channels)
        self.vmp = MPBlock(channels)
        self.ffn_h = FFN(channels, ffn_ratio)
        self.ffn_v = FFN(channels, ffn_ratio)

        # Eq. 4, cascaded OMP (selected by Table 6). "DS denotes a 2D CNN with a
        # stride of 2 in the last dimension."
        # INFERRED (guess_id: ffn-and-ds-widths): kernel size, bias and norm are
        # not stated, and Eq. 4 writes both operands as R^{C x ...} even though
        # consecutive ResNet stages differ in channel count, so a projection is
        # required that the paper never describes. The paper credits LGD (Qiu et
        # al. 2019) for the cascade; that author's own implementation
        # downsamples between stages with kernel 1, spatial stride 2, no bias,
        # plus BatchNorm. DS follows it and absorbs the channel projection.
        if prev_channels is None:
            self.ds_h = self.ds_v = None
        else:
            self.ds_h = nn.Sequential(
                nn.Conv2d(prev_channels, channels, kernel_size=1, stride=(1, 2), bias=False),
                nn.BatchNorm2d(channels),
            )
            self.ds_v = nn.Sequential(
                nn.Conv2d(prev_channels, channels, kernel_size=1, stride=(1, 2), bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(
        self,
        x: torch.Tensor,
        previous: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, C, T, H, W) -> (x_h, x_v) shaped (N, C, T, W) and (N, C, T, H)."""
        # Eq. 1 with AvgPool (Table 7): to isolate horizontal motion, average
        # away H; to isolate vertical motion, average away W.
        x_h = x.mean(dim=3)
        x_v = x.mean(dim=4)

        # Eq. 4: fuse the previous stage's decoupled features, downsampled.
        # The two orientation branches cascade independently (h->h, v->v), per
        # Fig. 3(b).
        if previous is not None and self.ds_h is not None:
            prev_h, prev_v = previous
            x_h = self.ds_h(prev_h) + x_h
            x_v = self.ds_v(prev_v) + x_v

        # Eq. 2 then Eq. 3, each with a residual (the FFN residual is drawn in
        # Fig. 3 but not written in any equation).
        x_h = self.hmp(x_h) + x_h
        x_v = self.vmp(x_v) + x_v
        x_h = self.ffn_h(x_h) + x_h
        x_v = self.ffn_v(x_v) + x_v
        return x_h, x_v


class OLMDBackbone(nn.Module):
    """ResNet34 with LMA, OMP and both coupling mechanisms after stages 2, 3, 4."""

    def __init__(self, context: int = 9, ffn_ratio: int = 4, coupling_width: int = 512):
        super().__init__()
        net = resnet34(weights="IMAGENET1K_V1")  # paper: ImageNet-pretrained ResNet34
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.stages = nn.ModuleList([net.layer2, net.layer3, net.layer4])

        self.lma = nn.ModuleList([LMA(c, context) for c in STAGE_CHANNELS])
        self.omp = nn.ModuleList(
            [
                OMP(c, prev_channels=prev, ffn_ratio=ffn_ratio)
                for c, prev in zip(STAGE_CHANNELS, (None,) + STAGE_CHANNELS[:-1])
            ]
        )

        # Eq. 5, stage coupling: X = X + alpha * US(X_h) + beta * US(X_v), with
        # alpha and beta "learnable scalars initialized to zero to preserve
        # original features". This matches the lineage exactly, where fusion is
        # x + module(x) * alpha[i] with alpha = nn.Parameter(torch.zeros(3)).
        self.alpha = nn.Parameter(torch.zeros(len(STAGE_CHANNELS)))
        self.beta = nn.Parameter(torch.zeros(len(STAGE_CHANNELS)))

        # Eq. 6, cross-stage coupling: X_CS = sum_i W_i (P_W(X_h^i) + P_H(X_v^i)),
        # where W_i is "the i-th fully connected layer".
        # INFERRED (guess_id: cross-stage-injection-point): the summands have
        # different channel counts per stage, so each W_i must project to a
        # shared width that the paper never states; `coupling_width` is that
        # width, set to the trunk's output width so X_CS can be added to the
        # frame features. Where X_CS enters the network is likewise never
        # stated, though Table 8 attributes 1.2 Dev / 1.3 Test WER to coupling.
        # It is added to the frame-level features feeding the temporal head,
        # which is the only place a (C, T) tensor fits Fig. 2's data flow.
        self.cross_stage = nn.ModuleList(
            [nn.Linear(c, coupling_width) for c in STAGE_CHANNELS]
        )
        self.coupling_width = coupling_width

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, C, T, H, W) -> frame features (N, coupling_width, T) and X_CS."""
        n, _, t = x.shape[:3]

        # Stem and stage 1 are per-frame 2D operations.
        flat = x.transpose(1, 2).reshape(n * t, *x.shape[1:2], *x.shape[3:])
        flat = self.stem(flat)
        flat = self.layer1(flat)

        previous = None
        cross_terms = []
        for index, (stage, lma, omp) in enumerate(zip(self.stages, self.lma, self.omp)):
            flat = stage(flat)

            # Back to (N, C, T, H, W) for the temporal modules.
            c, h, w = flat.shape[1:]
            volume = flat.reshape(n, t, c, h, w).transpose(1, 2)

            volume = lma(volume)  # Eqs. 7-11
            x_h, x_v = omp(volume, previous)  # Eqs. 1-4
            previous = (x_h, x_v)

            # Eq. 5: upsample by repeating features at each position, then fuse.
            up_h = x_h.unsqueeze(3).expand(-1, -1, -1, h, -1)
            up_v = x_v.unsqueeze(4).expand(-1, -1, -1, -1, w)
            volume = volume + self.alpha[index] * up_h + self.beta[index] * up_v

            # Eq. 6: pool each decoupled component to (N, C, T) and project.
            pooled = x_h.mean(dim=3) + x_v.mean(dim=3)
            cross_terms.append(self.cross_stage[index](pooled.transpose(1, 2)).transpose(1, 2))

            flat = volume.transpose(1, 2).reshape(n * t, c, h, w)

        frame_features = flat.mean(dim=(2, 3)).reshape(n, t, -1).transpose(1, 2)
        x_cs = torch.stack(cross_terms, dim=0).sum(dim=0)
        return frame_features, x_cs


class LocalWeighting(nn.Module):
    """TLP's local weighting: x + x * (sigmoid(InstanceNorm(conv(x))) - 0.5).

    Zero-initialized so it starts as the identity, matching TLP's released code.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x * (torch.sigmoid(self.norm(self.conv(x))) - 0.5)


class TemporalLiftPool(nn.Module):
    """Temporal Lift Pooling (TLP, Hu et al. 2022b), which the paper adopts by name.

    "We adopt the pooling method from TLP" is an explicit instruction, so the P2
    operator is TLP's learned lifting scheme rather than max pooling. Splitting
    the sequence into even and odd samples:

        d = X_odd - predictor(X_even)      the detail signal
        s = X_even + updater(d)            the approximation signal
        out = weight1(s) + weight2(d)

    The two auxiliary regularizers ||s - X_odd||_2 and ||d||_2 are returned and
    added to the objective with weight 0.001 each, matching TLP's released
    configuration (Cu, Cp). Both predictor/updater output convolutions are
    zero-initialized so pooling begins near a plain even-sample decimation.

    Note the detail branch is kept, not discarded: TLP's output sums a weighted
    approximation and a weighted detail signal.
    """

    def __init__(self, channels: int, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.predictor = self._branch(channels)
        self.updater = self._branch(channels)
        nn.init.zeros_(self.predictor[2].weight)
        nn.init.zeros_(self.updater[2].weight)
        self.weight1 = LocalWeighting(channels)
        self.weight2 = LocalWeighting(channels)

    @staticmethod
    def _branch(channels: int) -> nn.Sequential:
        # Depthwise k=3, ReLU, pointwise k=1, Tanh -- TLP's released structure.
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = x.shape[2]
        even = x[:, :, 0:length:self.kernel_size]
        odd = x[:, :, 1:length:self.kernel_size]
        # An odd-length sequence leaves `even` one sample longer; align them.
        if even.shape[2] != odd.shape[2]:
            even = even[:, :, : odd.shape[2]]
        detail = odd - self.predictor(even)
        approximation = even + self.updater(detail)
        loss_u = torch.norm(approximation - odd, p=2)
        loss_p = torch.norm(detail, p=2)
        return self.weight1(approximation) + self.weight2(detail), loss_u, loss_p


class TemporalHead(nn.Module):
    """Two 1D-CNN branches {K5,P2,K5,P2} plus a BiLSTM, with shared classifiers.

    The paper states the kernel configuration, that the two 1D-CNNs "share
    architecture", that TLP pooling is adopted, that the BiLSTM hidden size is
    1024, and that all classifiers are shared before the CTC losses (Eq. 12,
    following SMKD). Eqs. 12-13 name three CTC terms (local1, local2, global)
    and two distillation terms, which is why there are two 1D-CNN branches.

    INFERRED (guess_id: temporal-branch-topology): the paper does not say
    whether the two branches are sequential or parallel. They must be PARALLEL.
    Each {K5,P2,K5,P2} branch reduces length by roughly 4x, so two sequential
    branches reduce it ~16x: a 100-frame clip yields 2 output steps and a
    200-frame clip yields 8. PHOENIX14 training utterances average 11.5 glosses
    (p95 18, max 28), so CTC could not emit the target sequence -- and a
    T=32 probe crashed with "max_pool1d() Invalid computed output size: 0".
    One branch yields 22-72 steps for 100-300 frame clips, which is feasible.
    Both branches therefore read the same backbone features, matching "share
    architecture" and Eq. 12's two peer local losses.

    INFERRED (guess_id: bilstm-layer-count): the lineage's BiLSTM divides the
    given hidden size by the number of directions, so 1024 means 512 per
    direction with a 1024-wide output, and the convention is 2 layers. Fig. 2's
    "xN" annotation on the 1D-CNN blocks is never defined; N = 1 is used.

    The P2 operator is Temporal Lift Pooling, which the paper adopts by name
    ("We adopt the pooling method from TLP"), not max pooling. Its two auxiliary
    regularizers are returned and enter the objective with weight 0.001 each,
    matching TLP's released configuration.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        # Parallel peers over the same input, not a cascade (see the class docstring).
        self.branch1 = self._make_branch(input_size, hidden_size)
        self.branch2 = self._make_branch(input_size, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # per-direction, so output stays 1024
            num_layers=2,
            bidirectional=True,
        )
        # Eq. 12: "we share all classifiers before computing CTC losses".
        self.classifier = nn.Linear(hidden_size, num_classes)

    @staticmethod
    def _make_branch(input_size: int, hidden_size: int) -> nn.ModuleList:
        # {K5, P2, K5, P2} with TLP as P2. padding=0 on the convolutions,
        # matching the lineage, so each K5 shortens the sequence by 4; the
        # dataloader compensates with replicate padding.
        return nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=5),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(inplace=True),
                ),
                TemporalLiftPool(hidden_size),
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=5),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(inplace=True),
                ),
                TemporalLiftPool(hidden_size),
            ]
        )

    @staticmethod
    def _run_branch(branch: nn.ModuleList, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one {K5,P2,K5,P2} branch, accumulating TLP's auxiliary losses."""
        auxiliary = x.new_zeros(())
        for module in branch:
            if isinstance(module, TemporalLiftPool):
                x, loss_u, loss_p = module(x)
                auxiliary = auxiliary + loss_u + loss_p
            else:
                x = module(x)
        return x, auxiliary

    @staticmethod
    def update_lengths(lengths: torch.Tensor) -> torch.Tensor:
        """Sequence-length bookkeeping for one {K5,P2,K5,P2} branch.

        Each K5 has padding=0 and so removes 4 samples. TLP splits into even and
        odd samples and aligns them to the shorter, so it emits floor(L/2) --
        the same as MaxPool1d(2, ceil_mode=False).
        """
        for _ in range(2):
            lengths = lengths - 4
            lengths = torch.div(lengths, 2, rounding_mode="floor")
        return lengths

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> dict:
        local1, auxiliary1 = self._run_branch(self.branch1, features)
        local2, auxiliary2 = self._run_branch(self.branch2, features)
        lengths1 = self.update_lengths(lengths)
        lengths2 = lengths1  # parallel peers, so both share the reduced length

        # The BiLSTM does global temporal modelling over the local features.
        sequence = local2.permute(2, 0, 1)  # (T, N, C) for the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths2.cpu(), enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        global_feat, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        return {
            "local1_logits": self.classifier(local1.permute(2, 0, 1)),
            "local2_logits": self.classifier(local2.permute(2, 0, 1)),
            "global_logits": self.classifier(global_feat),
            "local1_lengths": lengths1,
            "local2_lengths": lengths2,
            "global_lengths": lengths2,
            # TLP's regularizers, weighted 0.001 each in the objective.
            "lift_pool_loss": auxiliary1 + auxiliary2,
        }


class OLMD(nn.Module):
    """Full OLMD model: backbone, temporal head, and the Eq. 14 loss."""

    def __init__(
        self,
        num_classes: int,
        context: int = 9,
        hidden_size: int = 1024,
        ffn_ratio: int = 4,
    ):
        super().__init__()
        self.backbone = OLMDBackbone(context, ffn_ratio, coupling_width=STAGE_CHANNELS[-1])
        self.head = TemporalHead(STAGE_CHANNELS[-1], hidden_size, num_classes)

    def forward(self, video: torch.Tensor, lengths: torch.Tensor) -> dict:
        frame_features, x_cs = self.backbone(video)
        # Cross-stage coupling injected into the frame features (see the
        # cross-stage-injection-point note in OLMDBackbone).
        return self.head(frame_features + x_cs, lengths)


# Eq. 14 loss weights, stated in the paper: "gamma_1 and gamma_2 are two
# hyperparameters used to balance contributions, we set them to 1 and 25,
# respectively." These are also exactly the lineage's released defaults
# (ConvCTC 1.0, SeqCTC 1.0, Dist 25.0), so they are an inherited convention
# rather than a tuned pair.
GAMMA_CTC = 1.0
GAMMA_DISTILL = 25.0

# TLP's auxiliary lifting regularizers. The OLMD paper adopts TLP's pooling by
# name but never restates these weights; TLP's released configuration uses
# Cu = Cp = 0.001, applied to the update and prediction norms respectively.
# Recorded as guess tlp-auxiliary-weights.
GAMMA_LIFT = 0.001


class OLMDLoss(nn.Module):
    """Eqs. 12-14: three CTC terms plus two self-distillation terms.

    INFERRED (guess_id: distillation-temperature): Eq. 13 gives no temperature
    and does not say whether the teacher is detached. The VAC/lineage
    formulation this loss is credited to uses KLDivLoss(reduction='batchmean')
    on log_softmax(student/T) against softmax(teacher/T), scaled by T*T, with a
    detached teacher and the blank class optionally excluded. T = 8 and
    blank-excluded are the lineage's released settings.
    """

    def __init__(self, blank: int = 0, temperature: float = 8.0, exclude_blank: bool = True):
        super().__init__()
        # reduction="none" then .mean() over the batch, matching the lineage.
        # NOT reduction="mean": torch divides each sample by its TARGET LENGTH,
        # which for PHOENIX14's mean of 11.5 glosses shrinks the CTC gradient
        # about 11.5x relative to the distillation term. With gamma_2 = 25 that
        # let distillation dominate, and the three branches settled into a
        # self-consistent all-blank minimum -- exactly what
        # train-phoenix14-seed0-attempt1 exhibited: dev WER pinned at 100.0 with
        # 78.6% deletions for 29 epochs while the loss sat flat near 15.8.
        self.ctc = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.exclude_blank = exclude_blank

    def _distill(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        start = 1 if self.exclude_blank else 0
        t = self.temperature
        student_logp = F.log_softmax(student[:, :, start:] / t, dim=-1)
        teacher_prob = F.softmax(teacher[:, :, start:].detach() / t, dim=-1)
        flat = student_logp.reshape(-1, student_logp.shape[-1])
        return self.kl(flat, teacher_prob.reshape(-1, teacher_prob.shape[-1])) * t * t

    def forward(self, outputs: dict, targets: torch.Tensor, target_lengths: torch.Tensor) -> dict:
        ctc_total = 0.0
        for key in ("local1", "local2", "global"):
            logits = outputs[f"{key}_logits"]
            ctc_total = ctc_total + self.ctc(
                F.log_softmax(logits, dim=-1),
                targets,
                outputs[f"{key}_lengths"],
                target_lengths,
            ).mean()

        # Eq. 13: the global branch teaches both local branches.
        teacher = outputs["global_logits"]
        distill = self._distill(outputs["local1_logits"], teacher) + self._distill(
            outputs["local2_logits"], teacher
        )

        lift = outputs.get("lift_pool_loss")
        lift_term = lift if lift is not None else targets.new_zeros((), dtype=torch.float32)

        return {
            "loss": (
                GAMMA_CTC * ctc_total
                + GAMMA_DISTILL * distill
                + GAMMA_LIFT * lift_term
            ),
            "ctc": ctc_total,
            "distill": distill,
            "lift": lift_term,
        }
