# Video-Based Sign Language Recognition via ResNet and LSTM Network

**Paper ID:** `990030f8dfefb06e99c05218741e11ccf7b08fdb`
**Citation:** Huang, J.; Chouvatut, V. *Journal of Imaging* 10(6), 149 (2024). https://doi.org/10.3390/jimaging10060149
**Preference level:** 3

Clean-room implementation; no author code was found.

**Pipeline status:** `insufficient_information`
**Numerical agreement:** `not_assessed` — no comparable target value exists because the paper's signer split is unresolved.

## Reproduction agents

**Original reproduction agent:** model unknown; application unknown. A 2026-09-10 audit of the [introducing commit](https://github.com/sign-language-processing/reproduction/commit/03f4581a4c9b252e410c714c153e578a35e7e2e4) and [PR #34](https://github.com/sign-language-processing/reproduction/pull/34) (including commits, comments, and reviews) found no execution attribution.

`reproduction.json.agents` preserves the evidence and its limits. Each historical run has `agent_ids: []` and an `agent_attribution_note`; no model or executor has been inferred from the human author or branch name.

## Target

The requested Table 4 contribution is the paper's ResNet-LSTM row: 86.25% accuracy. Table 3 identifies the same epoch-30, batch-16 result and also reports 84.98% F1 and 87.77% precision. These are the three targets. The other Table 4 rows are cited baselines and are out of scope.

| Metric | Paper (Table 3) | Conditional frozen-run result | Difference |
| --- | ---: | ---: | ---: |
| Accuracy | 86.25% | 76.2500% | -10.0000 |
| F1 score | 84.98% | 75.6829% | -9.2971 |
| Precision | 87.77% | 82.3100% | -5.4600 |

The conditional result is not entered as a reproduced target result: the paper does not identify its 8/2 signer split, so its numerical agreement with Table 3 cannot be assessed.

## What the paper specifies

| Item | Evidence |
| --- | --- |
| Model | ImageNet-pretrained ResNet-18 frame encoder, LSTM, then classifier (§§3.2, 3.5) |
| Frozen encoder | The pretrained hidden-layer parameters are retained/frozen (§4.4) |
| Video input | 16 RGB frames, 128×128, random crop (§§4.1, 4.4) |
| Data size | 3,200 clips; 2,560 train / 640 validation (§§4.2, 4.5) |
| Split description | 8 learners train, 2 learners test (§4.2); identities are not given |
| Target row | 30 epochs, batch 16 (§4.5, Table 3) |
| Optimizer/loss | Adam, LR 1e-4, weight decay 5e-4, label-smoothed cross entropy (§4.4) |
| Schedule | ReduceLROnPlateau, threshold 1e-4, patience 5, factor 0.1 (§4.4) |

The paper is internally inconsistent: §4.4 says Adam, while Table 2 says SGD; the reconstruction follows the detailed implementation prose. The paper describes 50-epoch experiments, but Table 3 explicitly reports the requested 30-epoch row.

## Guesses and deviations

These choices are not presented as paper facts:

| ID | Kind | Description | Paper says | This attempt used | Basis | Effect | Scope | Status | Evidence | Related IDs | Affects | Dataset IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `signer-split` | `guess` | Which two learners are held out. | 8 learners train, 2 learners test (§4.2); identities are not given | Held-out signers `(5, 10)`, only an inferred 8/2 convention, not an author split |  | LSA64 says subject 10 was replaced between its two disjoint recording sessions. A result on this split is a conditional reconstruction, not a verified reproduction |  | Gate `paper-signer-split` open |  |  |  | `lsa64-original` |
| `frame-selection` | `guess` | Positions of the 16 frames taken from each clip. | Not specified | Uniform 16-frame sampling |  |  |  |  |  |  |  |  |
| `resize-policy` | `guess` | Resize policy before the random crop. | Not specified | Resize-short-side 144 then one 128 crop |  |  |  |  |  |  |  |  |
| `normalization` | `guess` | Input normalization constants. | Not specified | ImageNet normalization |  |  |  |  |  |  |  |  |
| `projection-and-lstm` | `guess` | Width of the modified fully-connected layer and LSTM, and LSTM depth. | Not specified | A trainable 512-unit frame projection followed by one 512-unit LSTM |  |  |  |  |  |  |  |  |
| `seed` | `guess` | Random seed. | Not specified | 2024 |  |  |  |  |  |  |  |  |
| `scheduler-monitor` | `guess` | Quantity monitored by ReduceLROnPlateau. | Not specified | Training loss | Training loss prevents the held-out clips from controlling optimization |  |  |  |  |  |  |  |
| `optimizer` | `paper_internal_conflict` | Optimizer. | §4.4 says Adam, while Table 2 says SGD | Adam | The reconstruction follows the detailed implementation prose |  |  |  |  |  |  |  |
| `epoch-count` | `paper_internal_conflict` | Number of training epochs. | The paper describes 50-epoch experiments, but Table 3 explicitly reports the requested 30-epoch row | Fixed 30 epochs |  |  |  |  |  |  |  |  |

## Noncompliance

| ID | Description | Requirement | Run IDs | Effect | Resolution |
| --- | --- | --- | --- | --- | --- |
| `unpermitted-checkpoint-upload` | The earlier, excluded fine-tuned checkpoint (`model.pt`, SHA-256 `d2f155cbf44b95322ce812159e476c7536988d416500ca5656190e54030ee8e0`) was public at https://huggingface.co/repro-sign/huang-chouvatut-2024-resnet-lstm (the file is unchanged since revision `7ee40aeea3e96042eaaeee5256017965d70c06f4`). Its card was relabelled from `cc-by-nc-sa-4.0` to `cc-by-nc-nd-4.0` on 2026-11-12 (revision `0e43f972f18aad2a2e6365dcb2d8a7ced9286924`) but still reports that run as `complete` with 99.6875% accuracy. | Large artifacts are published only when their license permits redistribution; LSA64 is CC BY-NC-ND 4.0 and gate `lsa64-license-and-publication` is open |  | It is not reproduction evidence, and the public model card contradicts this report | Repository made private on 2026-11-12 (anonymous access returns 401). It stays private, or is removed, until permission to publish learned weights is established at gate `lsa64-license-and-publication` |

## Data license and open gates

The official LSA64 page currently licenses the data CC BY-NC-ND 4.0 for strict academic use and prohibits derivative works. The user authorized this private project-cloud run; no dataset or predictions are committed or distributed. Weights from an earlier, excluded run were published on Hugging Face and have since been made private; see `unpermitted-checkpoint-upload` under Noncompliance.

The unresolved signer identities and license/publishing question are open gates, so no target-level numerical result is claimed here. Earlier fine-tuned and purportedly frozen results are excluded: the retained code did not enforce the paper's encoder freeze, and the fine-tuned run also drove its LR schedule from the reported holdout.

## Conditional full run

The structured run policy was backfilled from the pre-migration report (whose
SHA-256 is retained in `reproduction.json`). The committed 24-hour launcher
ceiling is preserved; exact process boundaries, declaration time, retry maximum,
GPU-hour ceiling, and cost ceiling remain explicit unknowns.

The corrected frozen-backbone recipe completed successfully on Modal `repro-sign`: one A10G, seed 2024, fixed 30 epochs and batch size 16. It froze ResNet-18 parameters and BatchNorm state, kept the modified 512-unit frame projection/LSTM/classifier trainable, scheduled only from training loss, and evaluated the inferred 640-clip holdout once at epoch 30. The terminal evidence is app `ap-JrUpXgewdIOzCRYscgQhQQ`, function call `fc-01M0SQ06EG6Q7XKQWTZYBC8NR0`, and container `ta-01M0SQ076YFW7VXP9X20ZFQD7R`.

Raw metrics: `metrics.jsonl` SHA-256 `d776f1db74429596f77c676fd5d4740643ad19e021ab115e4f1a81a0692dc169`; run manifest SHA-256 `81bde9d05e7c8616a11b3a605f17073ba31b9b9c7098e0b23a1fd1376a3c8b0b`. The epoch-30 checkpoint remains only in the private results Volume and is not published.

## Excluded fine-tuning observation

An earlier 30-epoch run on the same inferred split reached 99.6875% held-out accuracy when it **unfroze the ImageNet ResNet-18 encoder**. It is retained here solely as a diagnostic observation: it is not a paper-faithful setting, not a reproduced target result, and not evidence for numerical agreement. It conflicts with §4.4's instruction to retain the pretrained hidden-layer parameters; that earlier run also used the held-out score to control its learning-rate schedule. The frozen result above is therefore the only retained conditional reconstruction.

## How to run the conditional reconstruction

The user authorized this private project-cloud run. The split remains a conditional reconstruction, and the separate trained-weight publication gate remains open. Authenticate to Modal workspace `repro-sign` and run:

```bash
./setup.sh
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/huang-chouvatut-2024-resnet-lstm/modal_app.py::populate_lsa64
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --detach \
  papers/huang-chouvatut-2024-resnet-lstm/modal_app.py::train
```

The data command verifies/populates `datasets/lsa64`. Training builds from the root study image, which supplies `simple-video-utils==0.7.4`, uses the shared `huggingface-cache`, and writes only to the paper-specific Modal results Volume. It runs a fixed 30-epoch, batch-16 recipe; the final held-out evaluation is performed once, at epoch 30, with no best-checkpoint selection.

## Provenance

| Artifact | Source | Evidence | Use |
| --- | --- | --- | --- |
| Paper | [Publisher PDF](https://mdpi-res.com/d_attachment/jimaging/jimaging-10-00149/article_deploy/jimaging-10-00149-v2.pdf?version=1719278950) | SHA-256 `01306636261ad2be13436204deab1b0bf04c7a732a654d2f03587c5f97edb4bb` | Targets and protocol |
| Dataset | [Official LSA64 release](https://midusi.github.io/lsa64/) | Original archive SHA-256 `218197acaa188583c1f06d149750af6af0d6b2bd44a627550d55773f5eefb20e` | 3,200 original RGB MP4s |

All video decoding uses `simple-video-utils==0.7.4`. The dataset manifest on the shared `datasets` Volume records the source archive, paths, and 3,200-clip count.
