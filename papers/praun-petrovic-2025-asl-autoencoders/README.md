# ASL autoencoders (Praun-Petrovic et al., 2025) reproduction

**Paper ID:** `01da9a64404d009c2ab0f22113b6c29c91f325b9`

**Citation:** Vouk Praun-Petrovic, Aadhvika Koundinya, Lavanya Prahallad. *Comparison of Autoencoders for tokenization of ASL datasets.* arXiv:2501.06942v1 [cs.LG], 12 January 2025.

**Paper:** https://arxiv.org/abs/2501.06942v1 · **Code/artifacts:** none found after search (see Source provenance)

**Preference level:** 3

Level 3 means a reimplementation from the paper.

**Pipeline status:** `insufficient_information`

**Numerical agreement:** `not_assessed` — No target produced a comparable value. All three Table 2 rows depend on hyperparameters, an image subset and architecture details that the paper does not report. The conditional values are listed below for the reviewer.

**Attempt date:** 2026-10-14

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `opus-5-5-claude-code` | Opus 5.5 (`claude-opus-5-5`) | Claude Code 2.1.299 | Whole attempt: queue-record ingestion, target ledger, source search, dataset check, protocol gate, `train.py` and `modal_app.py`, preflights, the three conditional full runs, and this report | The session environment declares model 'Opus 5.5' and ID `claude-opus-5-5`; `claude --version` printed `2.1.299 (Claude Code)`. All three runs list this agent in `agent_ids`. |

## Scope and target contract

The queue record asks for "Table 2: MSE for Reconstructed Images Across Different Encoder-Decoder Models on Validation Dataset". Table 2 (p. 7, Section 6.0.2) has three rows, and each becomes one target. All three are the authors' own models, so no row is a copied baseline.

| Target ID | Row | Published MSE |
| --- | --- | ---: |
| `table2-mse-feedforward` | FeedForward | 0.00506 |
| `table2-mse-convolutional` | Convolution | 0.00144 |
| `table2-mse-diffusion` | Diffusion | 0.00141 |

**Metric.** Mean squared error between the reconstructed and the input image. Images are resized to 200x200 and scaled to [0, 1] (Section 2.2), and the MSE is averaged over the validation set (Section 5.1). Lower is better. The paper reports one number per model, with no seeds or repeats.

**Validation set.** The paper contradicts itself about the split:

- Section 2.1 describes an 80/20 split of all 87,000 images, giving 17,400 validation images.
- Section 4 says "500 images per class" were split 80/20.
- Section 5.1 says the validation set "consists of 2,900 images randomly selected from the dataset using a fixed random seed". That is 20% of 500 x 29.

Section 5.1 is the one that defines the MSE evaluation, so this attempt uses the 500-per-class protocol: 11,600 training and 2,900 validation images.

Table 1 (Mean Opinion Scores from five human raters) is not in the requested scope, and no human evaluation was carried out.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | https://arxiv.org/pdf/2501.06942v1 | `6b61452774017e5fdaa876eddd067565176351d7d193a46ea9dfd41eef7806a7` | Targets and protocol; v1 is the only version |
| Published code | none found | — | — |
| Study repository | https://github.com/sign-language-processing/reproduction | `939d8937bf2744a2dd2ac4c0825167797865bccd` (origin/main at branch creation) | Base for this attempt |
| Base image | `ghcr.io/sign-language-processing/reproduction` | `sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291` | NGC PyTorch environment |

**Source search (2026-10-14).** The queue lists `code_repos: N/A`. A search of the following places found no code, weights, supplement or project page:

- The PDF itself has no links or footnotes, and its references are background works and Wikipedia pages.
- The arXiv abstract page and API list only v1.
- Hugging Face has no paper page, models or Spaces citing arXiv:2501.06942.
- GitHub repository search for the exact title, "asl autoencoder diffusion" and the author names found nothing; user `Vouk` hosts an unrelated project.
- Web searches on the title and the author names returned only mirrors and summary sites.
- Zenodo has 0 records for the title.
- The Research Spark Hub website, the third author's affiliation, has only programme pages.

Preference level 3 is the only available path.

## Results

All values below are **conditional**: the runs used the guesses in the next-but-one section, and none of the targets is marked produced.

| Target ID | Paper location | System | Dataset/split | Metric | Original | Conditional run | Difference | Terminal reason / evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `table2-mse-feedforward` | Table 2, row 1 | Feedforward AE (grayscale, 82.5M params) | ASL Alphabet v1, 2,900-image validation | MSE, [0, 1] pixels | 0.00506 | 0.004883 | −0.000177 | `not_produced` / `protocol_ambiguous`; run `full-feedforward-1` |
| `table2-mse-convolutional` | Table 2, row 2 | Convolutional AE (10.65M params) | same | same | 0.00144 | 0.001776 | +0.000336 | `not_produced` / `protocol_ambiguous`; run `full-convolutional-1` |
| `table2-mse-diffusion` | Table 2, row 3 | Diffusion AE (16.07M params) | same | same | 0.00141 | 0.018703 | +0.017293 | `not_produced` / `protocol_ambiguous`; run `full-diffusion-1` |

**Selected blocker: `protocol_ambiguous`.** The data is complete and permitted, and the compute is trivial. What prevents every target is that the paper does not specify the procedure that produced its numbers. This applies to every row: training hyperparameters, the image subset and its seed. Some gaps are specific to one model:

- **Feedforward:** all layer widths are missing.
- **Diffusion:** almost every component is missing.
- **Convolutional:** this is the most fully specified model, but the paper's own description gives 10,651,139 parameters, not the stated "approximately 16 million". So the model that was run differs from the description in some unknown way.

**Observations for the reviewer.**
- **Model ordering:** in the conditional runs, convolutional is better than feedforward, which agrees with the paper. The diffusion model is the worst of the three, the reverse of the paper's ordering.
- **Why the diffusion result is so poor:** its training MSE flattened at about 0.019 from epoch 10 on. Under the guessed design, the decoder sees a latent estimated back from a noised one at a random timestep, so reconstruction quality depends on how well the noise is predicted. This follows from the guessed design; it says nothing about the authors' model, which is not described.
- **Diffusion parameter count:** the guessed diffusion model has 16,067,715 parameters. That is a conv encoder with separate mean and log-variance heads, a small FC denoiser and the conv decoder. This matches the paper's "approximately 16 million", which suggests the paper's figure belongs to the diffusion model and was repeated for the convolutional one.
- **Feedforward colour:** the feedforward MSE is on grayscale pixels, following Figure 1's 40,000-value input, while the other two use RGB.

## How to repeat this

```bash
./setup.sh  # once per clone
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh asl-alphabet manifest.json
for m in feedforward convolutional diffusion; do
  .agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
    papers/praun-petrovic-2025-asl-autoencoders/modal_app.py --model $m --run-name full-$m-1
done
```

- **What `modal_app.py` does:** it mounts the `datasets` Volume read-only at `/datasets`, mounts `huggingface-cache` at `/cache/huggingface` (unused by this pipeline), and writes to the Volume `01da9a64-asl-autoencoders-results/RUN_NAME/`.
- **Outputs per run:** `metrics.json` (validation MSE and the per-epoch log), `split.json` (exact train/validation filenames), `model.pt` (final weights), `pip-freeze.txt` and `nvidia-smi.txt`.
- **Run names:** `train.py` refuses to overwrite an existing run directory, so use a new run name to repeat a run.
- **Preflight:** append `--extra "--per-class 10 --epochs 2 --max-steps 3"`.
- **Resume:** not needed, since each full run takes about 5 minutes.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| ASL Alphabet | Kaggle `grassknoted/asl-alphabet` v1, `asl_alphabet_train`, all 29 classes | https://www.kaggle.com/datasets/grassknoted/asl-alphabet; Volume checked 2026-10-14 | GPL-2.0 per the source metadata; existing authorised project copy (queue `on_modal: yes`, also used by `silanon-2023-asl-cnn`) | `asl-alphabet/source/asl_alphabet_train/asl_alphabet_train` | 29 x 3,000 = 87,000 images, each class directory counted; test set 28 images. `manifest.json` `012e786c…`, source archive `7c572f14…`. Subset used: 500 per class, 11,600 / 2,900 (`split.json` `80d17c24…`) | The manifest's 28-class `training_classes` field belongs to another study and is not used. The 500-per-class subset is a guess (see below). |

## Environment and patches

- **Image:** `ghcr.io/sign-language-processing/reproduction@sha256:305b6165…`, with PyTorch 2.11.0a0+eb65b36914.nv26.02, torchvision 0.25.0a0, numpy 2.1.0 and pillow 12.1.1. The full freeze is `pip-freeze.txt` (sha256 `9c737efc…`), identical across all three runs.
- **Image pinning:** the runs referenced `:latest`, and ghcr.io resolved that tag to this digest when checked at launch. `modal_app.py` now pins the digest.
- **Hardware:** NVIDIA L4, driver 580.95.05, CUDA 13.0; fp32 with deterministic cuDNN.
- **Code hashes at launch:** `train.py` `69f5cde3…`, `modal_app.py` `995dcede…`. The only later edit was the digest pin.
- **Patches:** none. There is no upstream code to patch; everything is a reimplementation.

## Execution evidence

| Run ID / agent IDs | Attempt / max | Kind / targets | Platform / hardware | Seed/config | Start/end (UTC) | Exit / terminal state / reason | Failure class | Stop ceilings | Logs/artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full-feedforward-1` / `opus-5-5-claude-code` | 1 / 2 | conditional full / `table2-mse-feedforward` | Modal `repro-sign`/main, app [`ap-aHIDIIGEvmWGS1FOjfblXr`](https://modal.com/apps/repro-sign/main/ap-aHIDIIGEvmWGS1FOjfblXr), 1x L4 | seed 42, 20 epochs, batch 32, lr 1e-3 | 12:59:28 → 13:07:41 | 0 / succeeded / completed | — | 3 h, 1 GPU-h, CHF 5 | `full-feedforward-1/metrics.json` `e8e4dd39…` |
| `full-convolutional-1` / `opus-5-5-claude-code` | 1 / 2 | conditional full / `table2-mse-convolutional` | app [`ap-hgmBO8EYb5V8FSSEI7miLJ`](https://modal.com/apps/repro-sign/main/ap-hgmBO8EYb5V8FSSEI7miLJ), 1x L4 | same | 12:59:28 → 13:04:48 | 0 / succeeded / completed | — | same | `full-convolutional-1/metrics.json` `78e7b1d9…` |
| `full-diffusion-1` / `opus-5-5-claude-code` | 1 / 2 | conditional full / `table2-mse-diffusion` | app [`ap-AIfqNNBgiYXKc6ygEZfyIp`](https://modal.com/apps/repro-sign/main/ap-AIfqNNBgiYXKc6ygEZfyIp), 1x L4 | same | 12:59:28 → 13:05:24 | 0 / succeeded / completed | — | same | `full-diffusion-1/metrics.json` `103d3e8c…` |

- **Compute used:** about 0.33 GPU-hours in total, below CHF 1. Billing for the individual apps was not retrieved.
- **Peak GPU memory:** 2.0 GB (feedforward), 0.56 GB (convolutional), 0.69 GB (diffusion).
- **Artifact location:** all artifacts live in the Modal Volume `01da9a64-asl-autoencoders-results`; hashes are in `reproduction.json.artifacts`.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Image subset and split | 500 per class, 80/20; validation "randomly selected ... using a fixed random seed"; seed not given | `random.Random(42).sample` of 500 sorted filenames per class, then a `torch.randperm` (seed 42) 80/20 split of the 14,500 images | Simplest reading of Sections 4 and 5.1 | Behaviour-changing |
| Training hyperparameters | Adam, MSE loss, "multiple epochs", weights saved after training | lr 1e-3 (Adam default), batch 32, 20 epochs, final-epoch weights, fp32 | Common defaults | Behaviour-changing for all rows |
| Feedforward architecture | Figure 1: 40,000-value input, two hidden layers per side, 64-d latent | Grayscale input; 40000-1024-256-64-256-1024-40000, ReLU, sigmoid output | Follows Figure 1; widths invented | Behaviour-changing; MSE is on grayscale pixels |
| Convolutional decoder activation | ReLU on every (transposed) convolution, sigmoid output; linear layers unspecified | ReLU after the decoder linear layer only | Needed before the transposed convolutions | Minor; the 10.65M vs ~16M parameter gap is unexplained |
| Diffusion autoencoder | Mean/variance heads; 100-step beta schedule on the latent; FC denoiser "with five primary layers, multiple dynamically added layers, and final output layers" | Reparameterised 64-d latent; linear betas 1e-4 to 0.02; one random timestep per image; five 256-wide ReLU layers predicting the noise from (z_t, t/100); z0 recovered and decoded; loss = reconstruction MSE + noise MSE, no KL; evaluation uses the same stochastic forward pass with a fixed seed | Minimal model consistent with the text; "dynamically added layers" left out because they are undefined | Largest uncertainty; only the parameter count (16.07M vs ~16M) can be checked |
| MSE aggregation | MSE over the validation set | Sum of squared errors / number of elements | Equals the mean of per-image MSE for equal-size images | None expected |
| Conflicting split description | Section 2.1: 69,600/17,400; Sections 4 and 5.1: 500 per class, 2,900 validation | Sections 4 and 5.1 | Section 5.1 defines the reported metric | The paper's own inconsistency |

## Attempts, failures, and dead ends

- **Dataset check:** all 29 class directories hold 3,000 images, matching the manifest and Section 2.1.
- **Preflights:** one per model, using 10 images per class, 2 epochs of 3 steps, checkpoint save/reload and validation on 58 images. All three passed on an L4 (apps `ap-ddtIZb6jigPvH9ewtpQW4T`, `ap-svvQmfWHZK7ugemYISgRY0`, `ap-O4iA4WIZzpozF5hsToXXeW`). They were disposable and are not retained as runs; their outputs remain in the results Volume as `preflight-*-1`.
- **Full runs:** the three full runs succeeded on their first attempt. No retries, OOMs or build failures occurred.
- **Diffusion gap left alone:** the diffusion result is far from the paper. It was not tuned, because any change would be a new invented design aimed at the published number rather than a correction grounded in the paper.

## Candidate flags, ethics, and human evaluation

- **Confirmation:** the queue record has `status: final` but no `confirmation` field. The assigning user confirmed in the session that the record is approved; the validator still reports the missing field.
- **Queue export:** it stays local and is not committed. Reviewer email addresses in the preserved copy inside `reproduction.json` are replaced by role labels.
- **Human evaluation:** the flag `includes_human_evaluation: yes` refers to Table 1 (MOS from five raters). That table is outside the requested scope, so no participants were involved.
- **Ethics and copied scores:** `potential_ethical_concerns: no` and `copied_scores: no` both agree with the paper.

## Author and team contact

None. Now that this independent attempt is complete, contacting the authors is permitted, and it is the remaining way to obtain the hyperparameters, subset seed and model definitions. No contact was made.
