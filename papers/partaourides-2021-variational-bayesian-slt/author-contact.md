# Author-contact email — SENT

Status: drafted 2026-09-08; sent ~2026-09-17 (approximate — sent by the study author from a
personal mail client, exact date not recorded). No reply as of 2026-09-30; the reproduction
was closed on that basis. See `reproduction.json.author_contact`.

**To:** c.partaourides@cut.ac.cy; sotirios.chatzis@cut.ac.cy
**Cc:** ai.voskou@edu.cut.ac.cy; dkosmo@upatras.gr
**Subject:** Reproduction of Table 1 (arXiv:2102.06143) — request for training details

---

Dear Dr. Partaourides and colleagues,

We are carrying out an independent reproducibility study of sign-language-processing
research and are working on *"Variational Bayesian Sequence-to-Sequence Networks for
Memory-Efficient Sign Language Translation"* (ICPR 2020 Workshops / arXiv:2102.06143),
specifically Table 1 (Gloss2Text on RWTH-PHOENIX-Weather 2014T).

We could not locate a public code release for the paper, so we reimplemented the model
from the text (in PyTorch: 4+4 layer GRU encoder–decoder, 1000 units, no attention, with
the last encoder layer replaced by the Stick-Breaking GRU; Adam, lr 1e-5, batch 128,
dropout 0.2). Our results are far below the paper on every row, including the baseline,
and our stick-breaking variants do not train stably, so we suspect we are missing or
mis-setting details that the paper does not state. We would be very grateful for your
help on the points below.

## What we obtain (single seed, "until convergence" via early stopping on dev BLEU-4)

| Model | dev BLEU-4 | dev ROUGE-L | test BLEU-4 | test ROUGE-L | paper (test BLEU-4) |
|---|---|---|---|---|---|
| Baseline (GRU, no attention) | 4.7 | 19.4 | 3.8 | 16.8 | 16.3 |
| GRU_repar | 3.7 | 11.7 | 4.1 | 11.3 | 17.0 |
| GRU_bp | 3.4 | 12.5 | 3.7 | 11.9 | 17.0 |
| SB-GRU | 3.8 | 12.1 | 4.0 | 11.5 | 18.1 |

The baseline dev BLEU-4 rises to ~4.3 by epoch ~70 and then stays flat — it appears to
converge to this level rather than simply needing more epochs. For GRU_bp / SB-GRU with
our settings the IBP KL term is roughly two to three orders of magnitude larger than the
translation NLL, and the variational last encoder layer stops learning (dev BLEU-4 frozen
for 60+ epochs).

## 1. Baseline / general training

Values we used are in brackets.

- Is code for the paper (or an earlier related implementation) available anywhere? [we found none]
- Word-embedding dimension for the encoder and decoder? [1000]
- Encoder GRU direction — unidirectional or bidirectional? [unidirectional, 4 layers]
- "Without attention": how is the encoder connected to the decoder? We initialise each
  decoder layer from the corresponding final encoder-layer state and pass nothing else.
  Did you instead feed a fixed context vector at every decoder step, tie the last encoder
  state in some other way, or use the last encoder *output*? [layer-wise final state only]
- Learning-rate schedule — is lr held constant at 1e-5, or is there warmup / decay /
  plateau reduction? [constant 1e-5]
- What defines "until convergence" — a fixed number of epochs/steps, or early stopping on
  a dev metric (which one, what patience)? Roughly how many epochs did Table 1 models
  train for? [early stop on dev BLEU-4, patience 20 evals, ≤250 epochs]
- Checkpoint used for the reported numbers — last, or best dev (on which metric)? [best dev BLEU-4]
- Decoding: greedy or beam search; beam width; length/coverage penalty; max length? [beam 3, no penalty, max 50]
- Gradient clipping? [global norm 5.0]
- BLEU and ROUGE: which implementations/scripts and versions, and any tokenisation /
  lower-casing / ROUGE variant (we assumed ROUGE-L F1)? [sacrebleu 2.4.3, tok 13a, lower-cased; rouge-score ROUGE-L F1]
- Gloss side: any cleaning of the `orth` field (e.g. removing `__…__` markers, `-PLUSPLUS`,
  frequency cut-off)? Our training gloss vocabulary is 1085 types vs the paper's 1066.
  (Our German vocabulary matches exactly at 2887 / 1077 singletons.) [raw whitespace split, no cleaning]

## 2. Stick-Breaking GRU (Eqs. 1–15)

- Which weights are treated as "non-gate"? We apply the treatment only to the candidate
  weight `W_y` of Eq. 7 (the `ỹ_t` transform), only in the last encoder layer. Correct? [yes, W_y only, last encoder layer only]
- IBP innovation / strength hyperparameter **α** (Eq. 1)? [1.0]
- Inference cut-off threshold **τ** for omitting weights with q(z) < τ? [0.5]
- Gumbel-Softmax / concrete relaxation temperature **λ**: initial value, final value, and
  the annealing schedule/rate (you cite [19])? [1.0 → 0.5, linear over 20k steps]
- Number of Monte-Carlo samples used for the ELBO expectations? [1]
- Is the KL term weighted or annealed (β schedule, free bits, KL warm-up), and is it scaled
  per-dataset (KL/N) or per-minibatch? [linear KL warm-up 0→1 over 2000 steps; KL scaled by 1/N_train]
- Initialisation of the Kumaraswamy posterior parameters a_k, b_k? [a_k = 10, b_k = 1]
- Initialisation of the Gaussian weight-posterior standard deviation σ (Eq. 2)? [σ ≈ 1]
- Prior on W — is it the spherical N(w | 0, 1) of Eq. 2 exactly? [yes]
- For Eqs. 13–15 we use a single-sample MC estimate for KL[q(u)‖p(u)] and KL[q(W)‖p(W)]
  and an analytic Bernoulli KL for KL[q(z)‖p(z)] (for stability). Does that match your
  implementation?

## 3. Weight compression (the `,wc` rows and the "factor of 16" claim)

- What exactly is the "unit round off" procedure, and how is the target bit precision
  derived from the mean weight variance? Which tensors is it applied to (only the last
  encoder `W_y`, or the whole model)? Is it a floating-point mantissa truncation, a
  fixed-point / integer quantisation, or something else? [we used uniform per-matrix
  2-bit quantisation of the last encoder layer's effective W_y]

## 4. Table 1 itself

- Are the values single-run, or an average over seeds (how many)? Any std? [we assume single-run]

We would be happy to share our reimplementation and full logs, and of course to
acknowledge your assistance. Any pointers — even partial — would help us report the
paper fairly.

Thank you very much for your time.

Best regards,
[name]
REPRO-SIGN reproducibility study
