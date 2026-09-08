"""Gloss2Text SLT reimplementation for Partaourides et al. 2021 (arXiv:2102.06143).

Preference level 3: no published code. Architecture from Camgoz 2018 [10] (4+4 layer
GRU seq2seq, 1000 units, no attention). The last encoder layer is replaced by a
Stick-Breaking GRU whose candidate ("non-gate") weight W_y gets, depending on --mode:

    plain  : ordinary weight (Baseline row)
    repar  : full Gaussian posterior q(W)=N(mu, softplus(rho)^2), prior N(0,1)   [Eq. 15]
    bp     : IBP stick-breaking utility indicators Z masking W_y                  [Eqs. 13,14]
    sb     : both of the above (SB-GRU)

Every numeric choice the paper leaves unspecified is a module-level constant below,
grouped under GUESSES, and echoed into the run's results JSON.

Subcommands:
    preflight  tiny CPU smoke test of the whole path
    train      train one model, checkpoint best dev BLEU-4, write results JSON
    evaluate   score an existing checkpoint on dev+test (optionally --quantize-bits)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# GUESSES: values the paper does not state. Each is echoed into results JSON.
# --------------------------------------------------------------------------------------
GUESSES: dict[str, object] = {
    "embedding_dim": 1000,          # paper: "1000 units in each layer"; emb size not given
    "encoder_directionality": "unidirectional",
    "num_layers": 4,               # stated
    "hidden_units": 1000,          # stated
    "dropout": 0.2,               # stated
    "batch_size": 128,           # stated
    "adam_lr": 1e-5,            # stated
    "adam_betas": (0.9, 0.999),  # torch default; paper just says "ADAM [20]"
    "grad_clip_norm": 5.0,       # not stated
    "max_epochs": 400,           # "until convergence"
    "early_stop_patience": 20,   # epochs w/o dev BLEU-4 improvement
    "eval_every_epochs": 2,
    "label_smoothing": 0.0,      # not mentioned
    "beam_width": 3,             # Camgoz-style; not stated
    "max_decode_len": 50,
    "length_penalty": 0.0,
    "gloss_lowercase": False,    # glosses are upper-case tokens
    "text_lowercase": True,      # German translation column is already lower-case
    "min_token_freq": 1,        # keep all; paper vocab 1066 vs our 1085 (see README)
    "ibp_alpha": 1.0,           # innovation hyperparameter alpha; "controls sparsity"
    "cutoff_tau": 0.5,          # inference: drop weights with q(z) < tau
    "mc_samples": 1,            # "draw MC samples"; count not given
    "kl_warmup_steps": 2000,    # linear 0->1 ramp on the KL term
    "gumbel_tau_start": 1.0,    # Gumbel-Softmax / concrete temperature lambda
    "gumbel_tau_end": 0.5,
    "gumbel_anneal_steps": 20000,
    "kuma_ab_init": 1.0,        # initial Kumaraswamy a_k, b_k
    "kl_w_estimator": "montecarlo",  # paper says MC; closed-form also available
    "seed": 1,                  # no seed policy stated
}

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class Vocab:
    def __init__(self, tokens_iter, min_freq: int):
        from collections import Counter

        coun = Counter()
        for toks in tokens_iter:
            coun.update(toks)
        self.itos = list(SPECIALS)
        for tok, freq in sorted(coun.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                self.itos.append(tok)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks, add_bos_eos=True):
        ids = [self.stoi.get(t, UNK) for t in toks]
        return [BOS] + ids + [EOS] if add_bos_eos else ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i in (PAD, BOS):
                continue
            if i == EOS:
                break
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return out


def load_corpus(csv_path: Path):
    """Return list of (gloss_tokens, text_tokens) from a PHOENIX-2014-T corpus CSV."""
    pairs = []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            gloss = row["orth"].strip()
            text = row["translation"].strip()
            if GUESSES["gloss_lowercase"]:
                gloss = gloss.lower()
            if GUESSES["text_lowercase"]:
                text = text.lower()
            g = gloss.split()
            t = text.split()
            if g and t:
                pairs.append((g, t))
    return pairs


@dataclass
class Batch:
    src: torch.Tensor      # (B, Ts)
    src_len: torch.Tensor  # (B,)
    tgt_in: torch.Tensor   # (B, Tt)  <s> ...
    tgt_out: torch.Tensor  # (B, Tt)  ... </s>


def make_batches(pairs, sv: Vocab, tv: Vocab, batch_size: int, shuffle: bool, device):
    idx = list(range(len(pairs)))
    if shuffle:
        random.shuffle(idx)
    else:
        idx.sort(key=lambda i: len(pairs[i][0]))  # length-bucketed for eval speed
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        src = [sv.encode(pairs[i][0]) for i in chunk]
        tgt = [tv.encode(pairs[i][1]) for i in chunk]
        ms = max(len(s) for s in src)
        mt = max(len(t) for t in tgt)
        src_p = [s + [PAD] * (ms - len(s)) for s in src]
        tgt_p = [t + [PAD] * (mt - len(t)) for t in tgt]
        src_t = torch.tensor(src_p, dtype=torch.long, device=device)
        tgt_t = torch.tensor(tgt_p, dtype=torch.long, device=device)
        yield Batch(
            src=src_t,
            src_len=torch.tensor([len(s) for s in src], dtype=torch.long, device=device),
            tgt_in=tgt_t[:, :-1].contiguous(),
            tgt_out=tgt_t[:, 1:].contiguous(),
        )


# --------------------------------------------------------------------------------------
# Stick-Breaking GRU layer (last encoder layer only)
# --------------------------------------------------------------------------------------
def _softplus(x):
    return F.softplus(x) + 1e-6


class SBGRULayer(nn.Module):
    """One GRU layer with the variational treatment on the candidate weight W_y.

    Gates m_t, r_t use ordinary weights (paper: principles applied "only on the
    non-gate related weights"). h_t = (1-m)⊙h_{t-1} + m⊙ỹ_t  (paper Eqs. 3-7).
    """

    def __init__(self, in_dim: int, hid_dim: int, mode: str):
        super().__init__()
        assert mode in {"plain", "repar", "bp", "sb"}
        self.mode = mode
        self.in_dim, self.hid_dim = in_dim, hid_dim
        cat = in_dim + hid_dim

        # gate weights (ordinary)
        self.W_m = nn.Linear(cat, hid_dim)
        self.W_r = nn.Linear(cat, hid_dim)
        self.b_y = nn.Parameter(torch.zeros(hid_dim))

        # candidate weight W_y : (cat, hid)
        w0 = torch.empty(cat, hid_dim)
        nn.init.xavier_uniform_(w0)
        if mode in {"repar", "sb"}:
            self.W_y_mu = nn.Parameter(w0)
            self.W_y_rho = nn.Parameter(torch.full((cat, hid_dim), -5.0))  # small sigma
        else:
            self.W_y = nn.Parameter(w0)

        if mode in {"bp", "sb"}:
            ab = math.log(math.expm1(float(GUESSES["kuma_ab_init"])))  # softplus^-1
            self.kuma_a_raw = nn.Parameter(torch.full((hid_dim,), ab))
            self.kuma_b_raw = nn.Parameter(torch.full((hid_dim,), ab))
            # posterior Bernoulli logits per (cat, hid); init ~ p=0.9 retained
            self.z_logits = nn.Parameter(torch.full((cat, hid_dim), 2.2))

        self.last_kl = {"w": 0.0, "u": 0.0, "z": 0.0}

    # -- sampling helpers -------------------------------------------------------------
    def _sample_W(self):
        if self.mode in {"repar", "sb"}:
            sigma = _softplus(self.W_y_rho)
            eps = torch.randn_like(sigma)
            W = self.W_y_mu + sigma * eps
            if GUESSES["kl_w_estimator"] == "closedform":
                kl = 0.5 * (sigma.pow(2) + self.W_y_mu.pow(2) - 1.0 - 2.0 * torch.log(sigma))
                self.last_kl["w"] = kl.sum()
            else:  # Monte-Carlo, paper's stated approach
                log_q = (-0.5 * math.log(2 * math.pi) - torch.log(sigma)
                         - 0.5 * ((W - self.W_y_mu) / sigma).pow(2))
                log_p = -0.5 * math.log(2 * math.pi) - 0.5 * W.pow(2)
                self.last_kl["w"] = (log_q - log_p).sum()
            return W
        return self.W_y

    def _sample_Z(self, gumbel_tau: float):
        # stick variables u_k ~ Kumaraswamy(a_k, b_k), inverse-CDF reparam (Eq. 9)
        a = _softplus(self.kuma_a_raw)
        b = _softplus(self.kuma_b_raw)
        X = torch.rand_like(a).clamp(1e-6, 1 - 1e-6)
        u = (1.0 - (1.0 - X).pow(1.0 / b)).pow(1.0 / a).clamp(1e-6, 1 - 1e-6)
        pi = torch.cumprod(u, dim=0)  # (hid,)  prior retention prob per column

        # KL[q(u)||p(u)]  (Eq. 13), p = Beta(alpha, 1), MC with sampled u
        alpha = float(GUESSES["ibp_alpha"])
        log_p_u = math.log(alpha) + (alpha - 1.0) * torch.log(u)
        log_q_u = (torch.log(a) + torch.log(b) + (a - 1.0) * torch.log(u)
                   + (b - 1.0) * torch.log1p(-u.pow(a)))
        self.last_kl["u"] = (log_q_u - log_p_u).sum()

        # relaxed Bernoulli mask via Gumbel-sigmoid, temp lambda
        q_logits = self.z_logits
        L = torch.rand_like(q_logits).clamp(1e-6, 1 - 1e-6)
        logistic = torch.log(L) - torch.log1p(-L)
        z = torch.sigmoid((q_logits + logistic) / gumbel_tau)

        # KL[q(z)||p(z)]  (Eq. 14): prior Bernoulli(pi_k), posterior Bernoulli(sigmoid(logits))
        q_p = torch.sigmoid(q_logits).clamp(1e-6, 1 - 1e-6)
        p_p = pi.unsqueeze(0).expand_as(q_p).clamp(1e-6, 1 - 1e-6)
        log_q_z = z * torch.log(q_p) + (1 - z) * torch.log1p(-q_p)
        log_p_z = z * torch.log(p_p) + (1 - z) * torch.log1p(-p_p)
        self.last_kl["z"] = (log_q_z - log_p_z).sum()
        return z

    def effective_W(self, gumbel_tau: float, train: bool):
        """W_y actually used this step, plus reset KL accumulators."""
        self.last_kl = {"w": torch.zeros((), device=self.b_y.device),
                        "u": torch.zeros((), device=self.b_y.device),
                        "z": torch.zeros((), device=self.b_y.device)}
        if train:
            W = self._sample_W()
            if self.mode in {"bp", "sb"}:
                W = W * self._sample_Z(gumbel_tau)
        else:  # inference: posterior mean + hard cutoff tau
            W = self.W_y_mu if self.mode in {"repar", "sb"} else self.W_y
            if self.mode in {"bp", "sb"}:
                mask = (torch.sigmoid(self.z_logits) >= float(GUESSES["cutoff_tau"])).float()
                W = W * mask
        return W

    def forward(self, x, gumbel_tau: float, train: bool):
        # x: (B, T, in_dim)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hid_dim)
        W_y = self.effective_W(gumbel_tau, train)
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            cat = torch.cat([h, xt], dim=-1)
            m = torch.sigmoid(self.W_m(cat))
            r = torch.sigmoid(self.W_r(cat))
            cat_c = torch.cat([r * h, xt], dim=-1)
            y_tilde = torch.tanh(cat_c @ W_y + self.b_y)
            h = (1 - m) * h + m * y_tilde
            outs.append(h)
        return torch.stack(outs, dim=1), h  # (B,T,H), (B,H)

    def kl_terms(self):
        return self.last_kl["w"], self.last_kl["u"], self.last_kl["z"]


# --------------------------------------------------------------------------------------
# Seq2seq (no attention)
# --------------------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, mode: str):
        super().__init__()
        H = int(GUESSES["hidden_units"])
        E = int(GUESSES["embedding_dim"])
        L = int(GUESSES["num_layers"])
        p = float(GUESSES["dropout"])
        self.H, self.L, self.mode = H, L, mode

        self.src_emb = nn.Embedding(src_vocab, E, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, E, padding_idx=PAD)
        self.drop = nn.Dropout(p)

        # encoder: first L-1 layers are stock GRU, last layer is SBGRULayer
        self.enc_lower = nn.GRU(E, H, num_layers=L - 1, batch_first=True,
                                dropout=p if L - 1 > 1 else 0.0)
        self.enc_top = SBGRULayer(in_dim=H, hid_dim=H, mode=mode)

        self.dec = nn.GRU(E, H, num_layers=L, batch_first=True,
                          dropout=p if L > 1 else 0.0)
        self.out = nn.Linear(H, tgt_vocab)

    def encode(self, src, gumbel_tau: float, train: bool):
        x = self.drop(self.src_emb(src))
        low, h_low = self.enc_lower(x)          # h_low: (L-1, B, H)
        top_seq, h_top = self.enc_top(self.drop(low), gumbel_tau, train)  # (B,T,H),(B,H)
        h = torch.cat([h_low, h_top.unsqueeze(0)], dim=0)  # (L, B, H)
        return h

    def forward(self, batch: Batch, gumbel_tau: float, train: bool):
        h0 = self.encode(batch.src, gumbel_tau, train)
        y = self.drop(self.tgt_emb(batch.tgt_in))
        dec_out, _ = self.dec(y, h0.contiguous())
        logits = self.out(dec_out)
        return logits

    def kl_terms(self):
        return self.enc_top.kl_terms()


# --------------------------------------------------------------------------------------
# Decoding (beam search, no attention -> decoder state carries everything)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def beam_decode(model: Seq2Seq, batch: Batch, beam: int, max_len: int, lp: float):
    model.eval()
    device = batch.src.device
    B = batch.src.size(0)
    h0 = model.encode(batch.src, gumbel_tau=float(GUESSES["gumbel_tau_end"]), train=False)  # (L,B,H)

    results = []
    for b in range(B):
        h = h0[:, b:b + 1, :].repeat(1, beam, 1).contiguous()  # (L, beam, H)
        seqs = torch.full((beam, 1), BOS, dtype=torch.long, device=device)
        scores = torch.full((beam,), -1e9, device=device)
        scores[0] = 0.0
        done = [False] * beam
        finished = []
        for _ in range(max_len):
            last = seqs[:, -1:]
            y = model.tgt_emb(last)
            dec_out, h_new = model.dec(y, h)
            logp = F.log_softmax(model.out(dec_out[:, -1, :]), dim=-1)  # (beam, V)
            V = logp.size(-1)
            cand = scores.unsqueeze(1) + logp
            for i in range(beam):
                if done[i]:
                    cand[i, :] = -1e9
                    cand[i, EOS] = scores[i]
            flat = cand.view(-1)
            top_scores, top_idx = flat.topk(beam)
            beam_id = top_idx // V
            tok_id = top_idx % V
            seqs = torch.cat([seqs[beam_id], tok_id.unsqueeze(1)], dim=1)
            h = h_new[:, beam_id, :].contiguous()
            scores = top_scores
            new_done = []
            for i in range(beam):
                d = done[beam_id[i]] or (tok_id[i].item() == EOS)
                new_done.append(d)
            done = new_done
            if all(done):
                break
        # pick best by length-penalised score
        best_i, best_s = 0, -1e18
        for i in range(beam):
            length = seqs[i].size(0)
            pen = ((5 + length) / 6) ** lp if lp > 0 else 1.0
            s = scores[i].item() / pen
            if s > best_s:
                best_s, best_i = s, i
        results.append(seqs[best_i].tolist())
    return results


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def score_corpus(hyps: list[list[str]], refs: list[list[str]]):
    import sacrebleu
    from rouge_score import rouge_scorer

    hyp_str = [" ".join(h) for h in hyps]
    ref_str = [" ".join(r) for r in refs]
    metric = sacrebleu.BLEU(tokenize="13a", lowercase=True)
    bleu = metric.corpus_score(hyp_str, [ref_str])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rl = [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyp_str, ref_str)]
    return {
        "bleu4": bleu.score,
        "bleu_signature": str(metric.get_signature()),
        "bleu_precisions": bleu.precisions,
        "rouge": 100.0 * sum(rl) / max(len(rl), 1),
    }


# --------------------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------------------
def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_vocabs(train_pairs):
    mf = int(GUESSES["min_token_freq"])
    sv = Vocab((g for g, _ in train_pairs), mf)
    tv = Vocab((t for _, t in train_pairs), mf)
    return sv, tv


def evaluate_split(model, pairs, sv, tv, device, beam=None):
    beam = beam or int(GUESSES["beam_width"])
    hyps, refs = [], []
    for batch in make_batches(pairs, sv, tv, 64, shuffle=False, device=device):
        seqs = beam_decode(model, batch, beam, int(GUESSES["max_decode_len"]),
                           float(GUESSES["length_penalty"]))
        for s in seqs:
            hyps.append(tv.decode(s))
    refs = [t for _, t in pairs]
    # pairs were length-sorted inside make_batches for src; realign by decoding order
    # -> rebuild refs in the same emitted order
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    refs = [pairs[i][1] for i in order]
    return score_corpus(hyps, refs)


def train(args):
    set_seed(int(GUESSES["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data_dir)
    train_pairs = load_corpus(data / "PHOENIX-2014-T.train.corpus.csv")
    dev_pairs = load_corpus(data / "PHOENIX-2014-T.dev.corpus.csv")
    test_pairs = load_corpus(data / "PHOENIX-2014-T.test.corpus.csv")
    if args.limit:
        train_pairs = train_pairs[: args.limit]
        dev_pairs = dev_pairs[: max(8, args.limit // 8)]
        test_pairs = test_pairs[: max(8, args.limit // 8)]
    sv, tv = build_vocabs(train_pairs)
    print(f"train={len(train_pairs)} dev={len(dev_pairs)} test={len(test_pairs)} "
          f"|src_vocab|={len(sv)} |tgt_vocab|={len(tv)}", flush=True)

    model = Seq2Seq(len(sv), len(tv), args.mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=float(GUESSES["adam_lr"]),
                           betas=tuple(GUESSES["adam_betas"]))
    N = len(train_pairs)
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=float(GUESSES["label_smoothing"]))

    max_epochs = args.max_epochs or int(GUESSES["max_epochs"])
    patience = int(GUESSES["early_stop_patience"])
    best_bleu, best_epoch, bad = -1.0, -1, 0
    ckpt_path = Path(args.out_dir) / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_nll = ep_kl = nb = 0.0
        for batch in make_batches(train_pairs, sv, tv, int(GUESSES["batch_size"]),
                                  shuffle=True, device=device):
            step += 1
            frac = min(1.0, step / max(1, int(GUESSES["gumbel_anneal_steps"])))
            g_tau = (GUESSES["gumbel_tau_start"]
                     + frac * (GUESSES["gumbel_tau_end"] - GUESSES["gumbel_tau_start"]))
            kl_beta = min(1.0, step / max(1, int(GUESSES["kl_warmup_steps"])))

            logits = model(batch, g_tau, train=True)
            nll = ce(logits.reshape(-1, logits.size(-1)), batch.tgt_out.reshape(-1))
            kw, ku, kz = model.kl_terms()
            kl = (kw + ku + kz) / N
            loss = nll + kl_beta * kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(GUESSES["grad_clip_norm"]))
            opt.step()
            ep_nll += nll.item(); ep_kl += float(kl); nb += 1

        msg = (f"epoch {epoch:3d} nll={ep_nll / nb:.4f} kl={ep_kl / nb:.4f} "
               f"g_tau={g_tau:.3f} kl_beta={kl_beta:.2f} t={time.time() - t0:.0f}s")
        if epoch % int(GUESSES["eval_every_epochs"]) == 0 or epoch == max_epochs:
            dev = evaluate_split(model, dev_pairs, sv, tv, device)
            msg += f"  DEV bleu4={dev['bleu4']:.2f} rouge={dev['rouge']:.2f}"
            history.append({"epoch": epoch, "dev": dev})
            if dev["bleu4"] > best_bleu:
                best_bleu, best_epoch, bad = dev["bleu4"], epoch, 0
                torch.save({"model": model.state_dict(), "sv": sv.itos, "tv": tv.itos,
                            "mode": args.mode, "guesses": GUESSES, "epoch": epoch}, ckpt_path)
                msg += "  *saved*"
            else:
                bad += 1
        print(msg, flush=True)
        if bad >= patience:
            print(f"early stop: no dev BLEU-4 gain for {patience} evals", flush=True)
            break

    # final eval from best checkpoint
    blob = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(blob["model"])
    dev = evaluate_split(model, dev_pairs, sv, tv, device)
    test = evaluate_split(model, test_pairs, sv, tv, device)
    result = {
        "mode": args.mode,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "wall_seconds": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "counts": {"train": len(train_pairs), "dev": len(dev_pairs), "test": len(test_pairs),
                   "src_vocab": len(sv), "tgt_vocab": len(tv)},
        "guesses": {k: (list(v) if isinstance(v, tuple) else v) for k, v in GUESSES.items()},
        "dev": dev,
        "test": test,
        "history": history,
    }
    out_json = Path(args.out_dir) / "results.json"
    out_json.write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps({"mode": args.mode, "dev": dev, "test": test}), flush=True)
    return result


def _quantize_mantissa(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform per-matrix quantisation to `bits` bits (weight-compression eval)."""
    if bits <= 0:
        return t
    lo, hi = t.min(), t.max()
    if hi <= lo:
        return t
    levels = (1 << bits) - 1
    q = torch.round((t - lo) / (hi - lo) * levels)
    return q / levels * (hi - lo) + lo


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = getattr(args, "limit", 0)
    blob = torch.load(args.checkpoint, map_location=device)
    global GUESSES
    GUESSES = {k: (tuple(v) if isinstance(v, list) else v) for k, v in blob["guesses"].items()}
    sv = Vocab.__new__(Vocab); sv.itos = blob["sv"]; sv.stoi = {t: i for i, t in enumerate(sv.itos)}
    tv = Vocab.__new__(Vocab); tv.itos = blob["tv"]; tv.stoi = {t: i for i, t in enumerate(tv.itos)}
    model = Seq2Seq(len(sv), len(tv), blob["mode"]).to(device)
    model.load_state_dict(blob["model"])

    if args.quantize_bits:
        with torch.no_grad():
            layer = model.enc_top
            W = layer.W_y_mu if blob["mode"] in {"repar", "sb"} else layer.W_y
            if blob["mode"] in {"bp", "sb"}:
                W = W * (torch.sigmoid(layer.z_logits) >= float(GUESSES["cutoff_tau"])).float()
            Wq = _quantize_mantissa(W.data, args.quantize_bits)
            if blob["mode"] in {"repar", "sb"}:
                layer.W_y_mu.data.copy_(Wq)
                layer.mode = "repar" if blob["mode"] == "repar" else "sb"
            else:
                layer.W_y.data.copy_(Wq)

    data = Path(args.data_dir)
    dev_pairs = load_corpus(data / "PHOENIX-2014-T.dev.corpus.csv")
    test_pairs = load_corpus(data / "PHOENIX-2014-T.test.corpus.csv")
    if limit:
        dev_pairs, test_pairs = dev_pairs[:limit], test_pairs[:limit]
    dev = evaluate_split(model, dev_pairs, sv, tv, device)
    test = evaluate_split(model, test_pairs, sv, tv, device)
    res = {"mode": blob["mode"], "quantize_bits": args.quantize_bits, "dev": dev, "test": test}
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / "eval.json").write_text(json.dumps(res, indent=2))
    print("EVAL " + json.dumps(res), flush=True)
    return res


def preflight(args):
    """Tiny CPU smoke test: real data + real loss + checkpoint + real eval + metric."""
    args.limit = 64
    args.max_epochs = 2
    GUESSES["eval_every_epochs"] = 1
    GUESSES["early_stop_patience"] = 99
    GUESSES["kl_warmup_steps"] = 5
    GUESSES["gumbel_anneal_steps"] = 10
    for mode in ["plain", "repar", "bp", "sb"]:
        print(f"\n=== preflight mode={mode} ===", flush=True)
        args.mode = mode
        args.out_dir = str(Path(args.scratch) / f"pf_{mode}")
        r = train(args)
        assert math.isfinite(r["dev"]["bleu4"]), f"{mode}: non-finite BLEU"
        ev = argparse.Namespace(checkpoint=str(Path(args.out_dir) / "best.pt"),
                                data_dir=args.data_dir, quantize_bits=2, limit=16,
                                out_dir=str(Path(args.out_dir) / "q"))
        evaluate(ev)
    print("\npreflight OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", required=True, choices=["plain", "repar", "bp", "sb"])
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=train)

    p = sub.add_parser("evaluate")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--quantize-bits", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=evaluate)

    p = sub.add_parser("preflight")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--scratch", default="/tmp/pf")
    p.set_defaults(func=preflight)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
