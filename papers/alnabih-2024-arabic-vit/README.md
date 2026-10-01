# Arabic sign language letters recognition using Vision Transformer

**TL;DR:** The full three-epoch reconstruction completed: **99.40799% accuracy (8,060/8,108)** versus the published **99.3%**.
**Decision:** No execution decision awaits human input. This is an eligible documented reconstruction; exact historical membership and training settings remain unknown.

**Pipeline status:** complete
**Numerical agreement:** does_not_agree
**Preference level:** 3

Alnabih, A. F. and Maghari, A. Y. (2024), Multimedia Tools and Applications 83:81725–81739. [Paper](https://doi.org/10.1007/s11042-024-18681-3). This 2026-10-01 attempt targets **Table 4**. The selected model is `google/vit-large-patch16-224-in21k`, established by section 4.2.1, not the base or DINO variant. Table 3 precision/recall/F1 and webcam examples are outside the requested table.

The real-data diagnostic loaded the pinned ViT-L weights, trained four steps, saved and reloaded model/optimizer/RNG state, executed another optimizer step, and evaluated 32 held-out examples. Its 4/32 correct (12.5%) is solely an engineering preflight metric, **not a reproduced Table 4 score**. That original attempt stopped before full training. Following explicit user authorization to make documented judgment calls, a separate full reconstruction completed three epochs and final-checkpoint evaluation.

Sections 3.2/4.1 and the authoritative dataset contain **54,049** images, while Table 2 gives **37,853 + 8,111 + 8,111 = 54,075**. We treat the latter counts as a reporting error and implement the stated 70/15/15 ratio over the real corpus: **37,834 training, 8,107 validation, 8,108 test** examples from one NumPy seed-42 permutation. This resolves the execution choice without inventing 26 examples or claiming the authors' historical membership.

Before observing full-run scores, we fixed the omitted training settings to ordinary defaults from [Transformers 4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/training_args.py): three epochs, batch eight, AdamW learning rate 5e-5, betas 0.9/0.999, epsilon 1e-8, no weight decay, linear learning-rate decay without warmup, and gradient clipping at norm 1. We use full-parameter float32 fine-tuning and select the final checkpoint. The paper's generic instruction to “freeze the last layer” does not identify an executable freezing configuration; we explicitly choose to train the newly initialized 32-class head and encoder, as in the successful preflight. Figure 7 does not label its horizontal-axis units, so it does not determine an epoch count. Validation is measured after each epoch; test examples are evaluated only after training, without model selection or tuning to the paper score.

This run is eligible as a **documented reconstruction of the requested target**. We initially classified unreleased split membership and training settings as conditional evidence, then reassessed that interpretation under the user's explicit instruction to resolve ordinary omissions. The authoritative corpus, selected model, stated split proportions and requested metric remain the same. The reassessment occurred during epoch two, before primary test evaluation; validation had been observed, but no configuration, seed, checkpoint rule or evaluation setting changed. Exact recovery of the historical experiment is not claimed.

| Table 4 row | Published accuracy | Reproduction |
|---|---:|---|
| Own ViT-L model | 99.3% | **99.40799%** (+0.10799 pp) |
| Latif et al. [19], CNN | 97.6% | Copied literature baseline |
| Alsaadi et al. [20], AlexNet | 94.81% | Copied literature baseline |
| Zakariah et al. [21], EfficientNetB4 | 98% | Copied literature baseline |
| Alnuaim et al. [13], ResNet50/MobileNetV2 | 95% | Copied literature baseline |
| Duwairi and Halloush [14], VGGNet | 97% | Copied literature baseline |

All six values have terminal target records. The five copied rows are preserved out of execution scope. Numerical agreement is assessed at the paper's one-decimal precision: 99.4% versus 99.3%. The exact difference is +0.10799 percentage points; this is not a scientific success/failure judgment. Separately, Table 3 swaps precision/F1 compared with the conclusion (99.3/99.4 versus 99.4/99.3); this does not affect Table 4's accuracy target.

The full 15-page PDF was obtained through the authenticated tracker after Springer and ResearchGate supplied only a preview. Its SHA-256 is `86832f502c9b905498a1bbb930d7d4174a9a1914b96012936a42be8a3e55144e`; it is not redistributed. Full paper references, supplements/availability statements, the author's institutional page, author/exact-title GitHub searches, and Zenodo/OSF searches did not recover author code, split files or trained weights. The institutional page was inaccessible. Consequently an independent diagnostic and full reconstruction were implemented; no generic repository was passed off as the authors' code.

The data is [ArSL2018/ArASL Mendeley v1](https://data.mendeley.com/datasets/y7pckrw6z2/1), CC BY 4.0, mirrored at `pain/ArASL_Database_Grayscale` revision `114709884276379a01e0722d71cd590c8ad3a05d`. The exact shared `datasets/arasl-database-grayscale` path and manifest were checked. The parquet SHA-256 `7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3`, 54,049 rows and 32 classes were verified inside the run. Its HF split called `train` is the entire corpus, not the paper training partition. The public license permits project-cloud copying/processing of these existing hand crops. No new participants or webcam data were collected; no license exception was inferred from the tracker ethics flag.

The tracker's linked dataset record `0rl07ms2n5hrk7h`, named “Arabic Sign Language ArSL dataset,” instead points to the distinct [Kaggle unaugmented Arabic-sign release](https://www.kaggle.com/datasets/sabribelmadoui/arabic-sign-language-unaugmented-dataset) with a blank license field. That lead was not used. The paper's sections 3.2/4.1 and reference 16 identify the canonical 54,049-image ArSL2018 corpus selected above; its identity and permission are verified independently. The mismatched tracker dataset record is not evidence that this canonical corpus is unavailable, and was not marked as stored on Modal.

Weights are pinned to `google/vit-large-patch16-224-in21k` revision `6074eaf2211423e928c93b93ef773d5da618aa7e`. The diagnostic uses seed 42, a NumPy permutation with 32 train/32 held-out images, RGB conversion and the model's pretrained processor, full-parameter float32 training, AdamW lr 5e-5 and batch 8. None of those missing author settings are claimed recovered. The randomly initialized 32-class head is expected. No tuning toward the published number was performed.

The Modal image starts from the prescribed repository GPU image and pins transformers 4.46.3, pyarrow 19.0.1 and Pillow 11.1.0. The resolved environment reports PyTorch `2.11.0a0+eb65b36914.nv26.02`, CUDA 13.1, driver 580.95.05 and one NVIDIA A100-SXM4-80GB. Native dependency freeze and hardware output are retained externally. The original diagnostics used immutable Modal image `im-VsmEEbnMoCD0ZCEe0jbT2Q`; their OCI digest was not exposed. The exact-loop preflight and full run use the pinned base `ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291` and observed Modal image `im-QYb8LEzeWRz0i8DEwfGPG4`. Shared canonical Volumes are v2: datasets read-only, huggingface-cache read-write at `/cache/huggingface`, with both required HF variables. Paper result storage is a separate v1 Volume `repro-0285c237-results`.

Two bounded diagnostics were retained, each with a 1,800-second / 0.5 GPU-hour / CHF5 ceiling declared before launch:

- [preflight-1](https://modal.com/apps/repro-sign/main/ap-MnO9XWQI6DX92jPHoVajZc): remote training/resume/evaluation succeeded in 22.94 seconds, but local result deserialization failed because `torch.__version__` was returned as a TorchVersion object and local torch is absent. Remote native evidence was preserved. This run is correctly classified as a failed command.
- [preflight-2](https://modal.com/apps/repro-sign/main/ap-h9s3z4AnBPieEhfRwEzsLw): returning a JSON string fixed that boundary; the complete command exited zero. Remote work took 18.63 seconds (about 0.00518 GPU-hours, excluding startup). Four measured training steps processed 18.73 images/sec, with peak allocated GPU memory 6,221,885,440 bytes. A checkpoint is about 3.64 GB.

The later [exact-preflight-v1](https://modal.com/apps/repro-sign/main/ap-RthHmV35hOXJnSmo9EjdSO) completed two 64-example epochs, 32-example validation/test evaluation and checkpoint restoration in 85.60 seconds of wrapper time. Training throughput was 30.51 then 69.48 examples/second; peak GPU allocation was 7,313,126,912 bytes. Its 3/32 diagnostic test accuracy is not a target score. This supported a conservative **1.3 GPU-hour / approximately USD 4** full-run forecast, with a predeclared **four GPU-hour / CHF 20** ceiling and no full restart.

The [full-three-epochs-v1](https://modal.com/apps/repro-sign/main/ap-RdF4Y4BNq6tsVhriLvV6lU) run completed with exit zero. Native script timestamps are `2026-10-01T13:05:13.194983+00:00` to `2026-10-01T13:33:26.431532+00:00`; function call `fc-01M3VS0YV0HYCW6CZKTHJNHV8M` and task `ta-01M3VS2N4Y7DC3GG4NS2F8N1WR` identify the retained execution. Wrapper wall time was **1708.21 seconds (0.47450 GPU-hours)**, including subprocess startup and final artifact hashing. Native training/evaluation wall time was 1693.24 seconds; peak GPU allocation was 7,314,068,480 bytes. At the [observed Modal A100 80GB rate](https://modal.com/pricing) of USD 0.000694/second, the GPU-only estimate is **USD 1.19**, excluding CPU, memory, startup billing and credits. Actual billed CHF cost is unavailable.

| Epoch | Online training accuracy | Validation accuracy | Training time |
|---|---:|---:|---:|
| 1 | 95.57805% | 98.72949% | 480.50 s |
| 2 | 99.22292% | 99.18589% | 471.11 s |
| 3 | 99.72247% | 99.54360% | 471.68 s |

The final test used all **8,108** held-out examples once, producing **8,060 correct (99.40799%)**. Local verification independently recomputed this value from the saved prediction file, checked its hash against native output, and verified that the saved train/validation/test indices form a disjoint union of all 54,049 examples. The training script hash matches the script that emitted the metrics. There was no full-run failure, restart, early stopping, checkpoint selection by test accuracy or score-driven adjustment.

Every native checkpoint, metrics file (including predictions/labels/index membership), dependency freeze and hardware log has an external URI and SHA-256 in `reproduction.json`. The first evidence collection overlapped a checkpoint write and is excluded; final hashes were collected after both jobs completed. An attempted evidence read incorrectly requested v2 for the existing v1 results Volume; that read failed without modifying data and was corrected to v1. All cloud operations used the `repro-sign` wrapper. No author contact occurred.

Repeat from repository root after `./setup.sh` and authenticated `repro-sign` setup. Use a new run ID for an independent repeat; an existing run resumes its saved epoch checkpoint or returns terminal metrics without overwriting evidence. The default entry point retains the original `preflight-2` diagnostic:

```bash
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh arasl-database-grayscale manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/alnabih-2024-arabic-vit/modal_app.py
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/alnabih-2024-arabic-vit/modal_app.py::evidence
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-0285c237-results preflight-2/metrics.json metrics.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run -d papers/alnabih-2024-arabic-vit/modal_app.py --exact-preflight --run-id exact-preflight-repeat-001
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run -d papers/alnabih-2024-arabic-vit/modal_app.py --full --run-id full-repeat-001
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/alnabih-2024-arabic-vit/modal_app.py::evidence --run-id full-repeat-001
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/alnabih-2024-arabic-vit
```

Setup, training, checkpoint verification and tiny evaluation are contained in the single Modal entry point. Data population is unnecessary because the exact licensed corpus already exists in the canonical Volume. The added full_train.py entry point implements the declared reconstruction; --exact-preflight tests its real optimizer/scheduler/checkpoint path and --full executes the fixed three-epoch schedule. The entry point resumes an existing epoch checkpoint and returns existing terminal metrics instead of overwriting them.

GPT-6 using Codex performed discovery, implementation, execution and reporting. These model/application names come from session instructions; exact model ID and harness version were not exposed. The tracker export and source hash are retained with operator emails redacted. The current tracker schema has final paper status, a distinct database ID and no confirmation field, so this is preserved as a direct user-authorized assignment without fabricating legacy queue confirmation.

An independent GPT-6/Codex reviewer (`compute-inventory-reviewer`) visually checked the paper on 2026-10-01 and confirmed that page 9 states 54,049 images while page 10 Table 2 prints 37,853 / 8,111 / 8,111, totaling 54,075. This is a printed inconsistency, not an OCR error or rounding of 54,049. Page 11 confirms the selected ViT-L model, but supplies no split identifier that resolves the discrepancy. The reviewer contributed no implementation changes or experiments. Attribution comes from an explicit session attestation; exact model ID/version and harness version were unavailable.

The root GPT-6/Codex agent (`root-loop-reviewer`) independently reviewed split boundaries, the optimizer/scheduler and checkpoint logic, protocol comparability before primary test, and tracker-data provenance. It did not execute these runs. Its explicit session attestation supplies model/application identity; exact model ID/version and harness version were not exposed.

After completion, the root reviewer independently confirmed all 54,049 split indices are unique and disjoint, prediction indices exactly match the 8,108 held-out examples, labels and predictions are valid class IDs, and the saved prediction/split hashes match native metrics. Its independent count was also 8,060 correct (99.40799210656142%).
