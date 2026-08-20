# Reproduction report: Sign Language Transformers

**Paper ID:** `camgoz-2020-slt`

**Citation:** Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, and Richard Bowden. 2020. Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation. CVPR.

**Paper:** [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.pdf) · **Code:** [neccam/slt](https://github.com/neccam/slt/tree/90588825f6229474bc19ac7a6b30ea3116635ba3) · **Weights:** [repro-sign/neccam-slt at 6ae275a](https://huggingface.co/repro-sign/neccam-slt/tree/6ae275aec44b59d129f22fab36a7120f05f94eb3)

**Preference level:** 2

**Status:** `reproduced`

**Attempt date:** 2026-08-20

## Scope and target contract

The direct assignment was to complete `repositories/neccam/slt`. The supplied repository configuration is the joint Sign2(Gloss+Text) system with recognition and translation loss weights both equal to 1. This resolves to the lambda_R=1.0, lambda_T=1.0 row in paper Table 4: dev WER 35.13, dev BLEU-4 21.73, test WER 33.75, and test BLEU-4 21.22.

The full train/dev/test splits, seed 42, development-BLEU checkpoint selection, development beam selection, and upstream metric implementations were used. The paper does not identify the Table 4 seed or aggregation; seed 42 comes from the published configuration. None of the four targets is a copied baseline.

## Source provenance

| Artifact | Canonical source | Pinned revision / SHA-256 | Role |
| --- | --- | --- | --- |
| Paper PDF | CVF Open Access | `ae08194f6e7e00f604bd3e623aa4866f7c1a71d6c4690ab8ae3f9f60857225a1` | Targets and protocol |
| Published code | `https://github.com/neccam/slt` | `90588825f6229474bc19ac7a6b30ea3116635ba3` | Model, training, search, and metrics |
| pami0 feature mirror | `https://huggingface.co/datasets/lavinal712/slt` | `d5c32f2cd1cf27a26083671532a32e75c98dbae3` | Train/dev/test precomputed features |
| Selected checkpoint | `https://huggingface.co/repro-sign/neccam-slt` | revision `6ae275aec44b59d129f22fab36a7120f05f94eb3`; `model.ckpt` SHA-256 `72f6cac1723463f9c7781051ab9bcd77c34ed3675637389bf09f0f6a4bc7d576` | Reproduced seed-42 weights |

The GitHub repository is the authors' published implementation. Its release list contains no weights. The feature URLs in its download script no longer resolve; the pinned Hugging Face mirror contains the same filenames and expected record structure, but no surviving author checksum allows cryptographic equivalence to be established.

## Results

| Target ID | Paper location | System | Dataset/split | Metric + version | Original | Reproduced | Difference | Evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `table-4-joint-dev-wer` | Table 4 | Sign2(Gloss+Text), 1/1 | PHOENIX14T dev | upstream corpus WER | 35.13 | 80.5444 | +45.4144 | `evidence/logs/full-result.json` |
| `table-4-joint-dev-bleu4` | Table 4 | Sign2(Gloss+Text), 1/1 | PHOENIX14T dev | bundled BLEU-4 / sacrebleu 1.4.4 | 21.73 | 11.3566 | -10.3734 | `evidence/logs/full-result.json` |
| `table-4-joint-test-wer` | Table 4 | Sign2(Gloss+Text), 1/1 | PHOENIX14T test | upstream corpus WER | 33.75 | 77.7413 | +43.9913 | `evidence/logs/full-result.json` |
| `table-4-joint-test-bleu4` | Table 4 | Sign2(Gloss+Text), 1/1 | PHOENIX14T test | bundled BLEU-4 / sacrebleu 1.4.4 | 21.22 | 11.1158 | -10.1042 | `evidence/logs/full-result.json` |

Development selected recognition beam 10 and translation beam 4 with alpha -1. The complete pipeline produced every target, which determines the `reproduced` status. Numerical agreement is separate: all four values differ substantially from Table 4, and this report does not interpret that difference as scientific success or failure.

## How to repeat this

From a fresh checkout with Modal authenticated to `repro-sign` and the shared v2 Volumes present:

```bash
./setup.sh
repositories/neccam/slt/scripts/data.sh
repositories/neccam/slt/scripts/dry_run.sh
repositories/neccam/slt/scripts/train.sh
repositories/neccam/slt/scripts/eval.sh
repositories/neccam/slt/scripts/publish.sh
```

`data.sh` is an idempotent, checksum-verifying population entry point. Training and evaluation mount `datasets` read-only at `/datasets` and `huggingface-cache` read-write at `/cache/huggingface`; outputs go to the v2 Volume `neccam-slt-results`. `train.sh` saves the selected checkpoint under `neccam-slt/full-seed-42`; `eval.sh` reloads it and writes the raw dev/test pickles. If evaluation is interrupted after training, rerun only `eval.sh`. The full target metrics are reached by the train plus evaluation commands.

## Data provenance and permissions

| Dataset | Version/subset/splits | Source and access date | License/permission and cloud-use basis | Path in Volume `datasets` | Counts / manifest / checksum | Deviations |
| --- | --- | --- | --- | --- | --- | --- |
| RWTH-PHOENIX-Weather 2014T | pami0 1024-D CNN features; train/dev/test | Official dataset page and pinned `lavinal712/slt` mirror, 2026-08-20 | CC BY-NC-SA 4.0; non-commercial research processing on project cloud | `rwth-phoenix-2014-t/features/` | 7,096 / 519 / 642; `evidence/data/features.json` | Public mirror substituted for dead author URLs; original-file equivalence unprovable |

The mirror files declare pickle protocol 5, although opcode inspection found no protocol-5-only opcodes. Python 3.7 cannot accept that declaration. `data.sh` preserves each source archive and creates a protocol-4 derivative by changing only the two-byte decompressed header; every later byte is identical. The derivatives loaded with the expected counts and 1024-D feature width.

No video file is decoded by this experiment: the published loader consumes precomputed frame features. The simple-video-utils decoding requirement is therefore not applicable to this recipe.

## Environment and patches

The final image uses `nvidia/cuda:11.4.3-devel-ubuntu20.04` at digest `sha256:5d81539e4f3fab923fac7599baaf44ac055a0e69a47d49aed7c9da1e499c9cba`; Miniconda installer SHA-256 is `4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e`. Python is 3.7.13, PyTorch is 1.7.1+cu110, the CUDA runtime reported by PyTorch is 11.0, and the full dependency manifest is `evidence/environment/pip-freeze.txt`. Modal image `im-MmlcOnE9uxKbTIvDRme4n2` contains the terminal evaluation recipe. The shared cache was mounted at `/cache/huggingface` with both required Hugging Face environment variables.

| Patch | Demonstrated failure | Hypothesis | Why necessary | Behavioral effect | Evidence |
| --- | --- | --- | --- | --- | --- |
| `patches/torch-1.7-integer-division.patch`, SHA-256 `2f56606c5c8c5788e288a89f5ef3e5d7e8cc4c19f57387af570aaa2719e2204c` | `index_select` rejected Float beam indices at translation beam size above one | Floor division restores PyTorch 1.4 integer division for non-negative flattened beam IDs | PyTorch 1.7 changed `Tensor.div` to true division | Dtype compatibility only; selected IDs are unchanged | dry app `ap-tiICw61YXIQNrukApq8OYV`; terminal eval `ap-NoGVX71SmRgWKGQFE4NftC` |

The Docker dependency substitutions keep the closest executable published API families: torch/torchvision/torchtext 1.4.0/0.5.0/0.5.0 became 1.7.1+cu110/0.8.2+cu110/0.8.1; unavailable tensorboard, tensorflow-estimator, and warmup-scheduler patch releases were replaced by compatible releases. No model architecture or optimization setting was changed.

## Execution evidence

| Run ID | Kind | Targets | Platform/workspace | Hardware | Seed/config | Start/end UTC | Exit/terminal state | Wall/GPU hours | Cost | Logs/artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dry-run-001` | real-data dry run | all four path checks | Modal `repro-sign`, app `ap-tiICw61YXIQNrukApq8OYV`, call `fc-01M0FBE9Q025M69M128EA7YRKF` | A100 40GB | 42 / `configs/dry.yaml` | 10:28:39–10:29:18 | 0 / succeeded | 0.0109 / 0.0109 | unavailable | dry manifest and Modal dashboard |
| `full-train-seed-42` | full training, failed inline eval | checkpoint for all targets | Modal `repro-sign`, app `ap-u2zrVrQVSLIUMdeqAXWpOC`, call `fc-01M0F839DY5M7PQKYGTW72VZMC` | A100 40GB | 42 / `configs/sign.yaml` | 09:30:05–10:22:34 | 1 / failed after training | 0.8747 / 0.8747 | unavailable | run manifest, `train.log`, step-6200 checkpoint |
| `full-eval-seed-42` | checkpoint evaluation | all four | Modal `repro-sign`, app `ap-NoGVX71SmRgWKGQFE4NftC`, call `fc-01M0FBGCGHMFBCJC484HM9XRG0` | A100 40GB then A100 80GB | 42 / `configs/sign.yaml` | 10:29:40–10:58:50 | 0 / succeeded after one preemption | 0.4861 / 0.4861 | unavailable | run manifest, raw result JSON, dev/test pickles |

Training ran 8,000 optimizer steps through epoch 37. The best checkpoint was step 6,200 at training-time greedy dev BLEU 10.67. LR changed from 0.001 to 0.0007 at step 5,300 and to 0.00049 at step 7,200. Peak observed training allocation was 4.7617 GiB. The evaluation's first container was preempted once; Modal automatically restarted the same function call, which then completed. Exact metrics were persisted by collector app `ap-FnyTI6to9CsPRlQZ4vwSP2`, call `fc-01M0FD7GE8BCRZ4B58P48DVRW2`.

## Guesses and deviations

| Detail | Paper/evidence says | This attempt used | Rationale | Effect on interpretation |
| --- | --- | --- | --- | --- |
| Seed | Paper does not identify Table 4 seed/aggregation | Published config seed 42, one run | Least-invasive supplied default | Seed variation is not measured |
| Feature artifact | Author URLs in repository | Pinned public mirror | Author URLs are dead | Mirror identity is a plausible but unverified mismatch source |
| Pickle protocol | Published Python 3.7 environment | Header-only protocol 4 derivative | Mirror declares unsupported protocol 5 | Payload preserved; serialization declaration differs |
| Framework | torch 1.4 / CUDA-era stack | torch 1.7.1+cu110 on A100 | Original torch hung on A100; 1.7 is the earliest proven A100 path | Framework numerical behavior may differ |
| Search grid | Paper says beams 0–10, alphas 0–2 | Supplied config beams 1–10, alphas -1–5 | Reproduce repository default without tuning | Reported metrics follow code artifact rather than prose grid |
| Minimum LR | Paper says `1e-6` | Supplied config `1e-7` | Preserve repository default | Scheduler protocol differs from paper text |
| Hardware | Not reported | One A100, fp32 | Available bounded project compute | Exact hardware reproducibility is unavailable |

## Attempts, failures, and dead ends

- A CUDA 11.4.0 base tag was no longer available; 11.4.3 was selected and later pinned by digest.
- Modal's control runtime could not provide Python 3.7 directly, so the author runtime was installed inside the container while Modal used a current control interpreter.
- An early environment attempt leaked the control `PYTHONPATH` into Python 3.7 and failed imports; the training subprocess now removes it. That change was kept after a CUDA forward/backward probe passed.
- The original torch 1.4/CUDA 10.1 stack did not make GPU progress on A100. The smallest proven A100-compatible family, torch 1.7.1+cu110, completed forward/backward and was kept.
- The mirror protocol declaration failed in Python 3.7. Header-only conversion was tested byte-for-byte after the header and kept.
- The first complete dry path trained and evaluated but the Modal control interpreter lacked NumPy for result reading. Score extraction was moved into the pinned author environment.
- Full training completed, then PyTorch 1.7 beam search failed with `expected scalar type Long but found Float`. The single-line patch was first tested with beam 2 / alpha 0 on 32/8/8 real samples, then evaluation was rerun from the saved checkpoint.
- The patched evaluation was preempted once during recognition beam search. Modal's automatic retry restarted it; no manual restart or protocol change was made.
- The first CPU collector tried to restore embedded tensors to CUDA. A scoped `map_location=cpu` hook was added to the local extractor and the cheap collector then persisted exact metrics. No evaluation was rerun.

No OOM, multi-node run, author-held checkpoint, or training restart occurred.

## Candidate flags, ethics, and human evaluation

This was a direct user assignment, not a queue candidate, so there were no queue comments, copied-score flags, or ethics flags to resolve. The attempt processed an existing licensed benchmark on the project cloud, did not enroll or interact with participants, and performed no new human evaluation. The dataset was handled under its non-commercial research terms and was not committed or uploaded with the model.

## Author and team contact

No authors were contacted. No Team S data-access request or Team R compute discussion was required: the data and license basis were available, and the plan stayed single-GPU and below 24 GPU-hours.

## Terminal account

All four targets are `produced` and point to `evidence/logs/full-result.json`, so the pipeline status is `reproduced`. The training attempt's inline evaluation failed, but its verified checkpoint was retained; the minimally patched evaluation then reached a terminal state and produced every requested dev/test value. The substantial numerical discrepancies, unverifiable mirror identity, one-seed scope, and environment/search deviations remain explicit for human scientific review.
