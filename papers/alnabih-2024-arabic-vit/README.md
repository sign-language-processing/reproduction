# Arabic sign language letters recognition using Vision Transformer

**Pipeline status:** insufficient_information
**Numerical agreement:** not_assessed
**Preference level:** 3

Alnabih, A. F. and Maghari, A. Y. (2024), Multimedia Tools and Applications 83:81725–81739. [Paper](https://doi.org/10.1007/s11042-024-18681-3). This 2026-10-01 attempt targets **Table 4**. The selected model is `google/vit-large-patch16-224-in21k`, established by section 4.2.1, not the base or DINO variant. Table 3 precision/recall/F1 and webcam examples are outside the requested table.

The real-data diagnostic loaded the pinned ViT-L weights, trained four steps, saved and reloaded model/optimizer/RNG state, executed another optimizer step, and evaluated 32 held-out examples. Its 4/32 correct (12.5%) is solely an engineering preflight metric, **not a reproduced Table 4 score**. No full training run was launched.

The scientific blocker is an explicit split inconsistency: sections 3.2/4.1 and the cited dataset contain **54,049** images, but Table 2 gives **37,853 + 8,111 + 8,111 = 54,075**. The paper gives a 70/15/15 ratio but no released membership or artifact resolving the extra 26 images. A conditional resplit can run, but cannot establish the original test-set result. The open question is whether Table 2 is a typo and which split actually generated 99.3%. Missing routine optimizer settings are recorded as diagnostic assumptions, not raised as questions.

| Table 4 row | Published accuracy | Reproduction |
|---|---:|---|
| Own ViT-L model | 99.3% | Not produced: conflicting dataset/split counts |
| Latif et al. [19], CNN | 97.6% | Copied literature baseline |
| Alsaadi et al. [20], AlexNet | 94.81% | Copied literature baseline |
| Zakariah et al. [21], EfficientNetB4 | 98% | Copied literature baseline |
| Alnuaim et al. [13], ResNet50/MobileNetV2 | 95% | Copied literature baseline |
| Duwairi and Halloush [14], VGGNet | 97% | Copied literature baseline |

All six values have terminal target records. The five copied rows are preserved out of execution scope. No numerical agreement is assessed. Separately, Table 3 swaps precision/F1 compared with the conclusion (99.3/99.4 versus 99.4/99.3); this does not affect Table 4's accuracy target.

The full 15-page PDF was obtained through the authenticated tracker after Springer and ResearchGate supplied only a preview. Its SHA-256 is `86832f502c9b905498a1bbb930d7d4174a9a1914b96012936a42be8a3e55144e`; it is not redistributed. Full paper references, supplements/availability statements, the author's institutional page, author/exact-title GitHub searches, and Zenodo/OSF searches did not recover author code, split files or trained weights. The institutional page was inaccessible. Consequently a small independent diagnostic was implemented; no generic repository was passed off as the authors' code.

The data is [ArSL2018/ArASL Mendeley v1](https://data.mendeley.com/datasets/y7pckrw6z2/1), CC BY 4.0, mirrored at `pain/ArASL_Database_Grayscale` revision `114709884276379a01e0722d71cd590c8ad3a05d`. The exact shared `datasets/arasl-database-grayscale` path and manifest were checked. The parquet SHA-256 `7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3`, 54,049 rows and 32 classes were verified inside the run. Its HF split called `train` is the entire corpus, not the paper training partition. The public license permits project-cloud copying/processing of these existing hand crops. No new participants or webcam data were collected; no license exception was inferred from the tracker ethics flag.

Weights are pinned to `google/vit-large-patch16-224-in21k` revision `6074eaf2211423e928c93b93ef773d5da618aa7e`. The diagnostic uses seed 42, a NumPy permutation with 32 train/32 held-out images, RGB conversion and the model's pretrained processor, full-parameter float32 training, AdamW lr 5e-5 and batch 8. None of those missing author settings are claimed recovered. The randomly initialized 32-class head is expected. No tuning toward the published number was performed.

The Modal image starts from the prescribed repository GPU image and pins transformers 4.46.3, pyarrow 19.0.1 and Pillow 11.1.0. The resolved environment reports PyTorch `2.11.0a0+eb65b36914.nv26.02`, CUDA 13.1, driver 580.95.05 and one NVIDIA A100-SXM4-80GB. Native dependency freeze and hardware output are retained externally. The immutable Modal image ID is `im-VsmEEbnMoCD0ZCEe0jbT2Q`; an OCI digest was not exposed and is not invented. Shared canonical Volumes are v2: datasets read-only, huggingface-cache read-write at `/cache/huggingface`, with both required HF variables. Paper result storage is a separate v1 Volume `repro-0285c237-results`.

Two bounded diagnostics were retained, each with a 1,800-second / 0.5 GPU-hour / CHF5 ceiling declared before launch:

- [preflight-1](https://modal.com/apps/repro-sign/main/ap-MnO9XWQI6DX92jPHoVajZc): remote training/resume/evaluation succeeded in 22.94 seconds, but local result deserialization failed because `torch.__version__` was returned as a TorchVersion object and local torch is absent. Remote native evidence was preserved. This run is correctly classified as a failed command.
- [preflight-2](https://modal.com/apps/repro-sign/main/ap-h9s3z4AnBPieEhfRwEzsLw): returning a JSON string fixed that boundary; the complete command exited zero. Remote work took 18.63 seconds (about 0.00518 GPU-hours, excluding startup). Four measured training steps processed 18.73 images/sec, with peak allocated GPU memory 6,221,885,440 bytes. A checkpoint is about 3.64 GB.

A conditional 70/15/15 split would contain approximately 37,834 training images. Measured throughput gives about 2,020 seconds (0.56 GPU-hours) per training epoch before validation, startup and checkpoint overhead. The paper does not state the exact training budget/hyperparameter schedule; no complete configuration or full-run cost estimate is claimed. Actual billed cost is unavailable. This estimate does not authorize a scientifically comparable full run while the split gate remains open.

Every native checkpoint, metrics file (including predictions/labels/index membership), dependency freeze and hardware log has an external URI and SHA-256 in `reproduction.json`. The first evidence collection overlapped a checkpoint write and is excluded; final hashes were collected after both jobs completed. An attempted evidence read incorrectly requested v2 for the existing v1 results Volume; that read failed without modifying data and was corrected to v1. All cloud operations used the `repro-sign` wrapper. No author contact occurred.

Repeat from repository root. The entry point returns existing `preflight-2` metrics when present, preserving retained evidence; on an empty results Volume it executes the diagnostic:

```bash
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh arasl-database-grayscale manifest.json
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/alnabih-2024-arabic-vit/modal_app.py
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/alnabih-2024-arabic-vit/modal_app.py::evidence
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh volume get repro-0285c237-results preflight-2/metrics.json metrics.json
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/alnabih-2024-arabic-vit
```

Setup, training, checkpoint verification and tiny evaluation are contained in the single Modal entry point. Data population is unnecessary because the exact licensed corpus already exists in the canonical Volume. Full training/evaluation entry points are intentionally not presented as verified while the conflicting split remains unresolved.

GPT-6 using Codex performed discovery, implementation, execution and reporting. These model/application names come from session instructions; exact model ID and harness version were not exposed. The tracker export and source hash are retained with operator emails redacted. The current tracker schema has final paper status, a distinct database ID and no confirmation field, so this is preserved as a direct user-authorized assignment without fabricating legacy queue confirmation.
