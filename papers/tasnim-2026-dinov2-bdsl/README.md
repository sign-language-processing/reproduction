# A vision transformer-based fine-tuned DINOv2 model for bangla sign language recognition

Syeda Anika Tasnim, Rifath Mahmud, Tanvir Ahmed, and Debajyoti Karmaker (2026), Multimedia Tools and Applications 85:163. [Publisher DOI](https://doi.org/10.1007/s11042-026-21376-6).

**Conclusion:** The assigned Table 5 comparative finding is reproduced. Our trained model reaches **99.2177% accuracy and 99.2168% macro F1**, above every reported Table 5 comparator for the respective metric. Scientific review accepted this as a valid reproduction on 2026-10-03; the exact published 99.7% values remain unmatched.

This attempt reconstructs Table 5's first row, ViT-S-BDSL accuracy and F1, both 99.7%. Table 3 specifies macro and weighted F1, which coincide on the balanced 60-image-per-class test set. The other systems in Table 5 are outside the assigned scope. No copied baseline is presented as a newly run result.

GPT-6 using Codex performed discovery, implementation, execution and reporting. The master GPT-6/Codex agent also assisted with official browser source discovery and acquisition. An independent GPT-6/Codex reviewer checked the split, architecture, transforms, optimization, final checkpoint and metrics without executing experiments. Exact model ID and harness version are not exposed by these sessions; attribution records this explicitly. The original tracker export hash and redacted expanded final paper record are preserved as direct user-authorized assignment provenance in reproduction.json; the new tracker has no legacy confirmation property. The authorized publisher PDF was inspected in full, hashed and retained outside Git.

## Recipe and sources

No target-author code or supplement was located through the full paper, exact-title and author/method searches, institutional/ORCID publication records or GitHub. The classification pipeline is reconstructed, but its backbone is the unmodified [official DINOv2 repository](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8), pinned at 7764ea0f912e53c92e82eb78a2a1631e92725fc8. Hub construction and attention source were inspected before execution. The model is distilled ViT-S/14 without registers, followed by linear 384→256, ReLU and linear 256→49; all layers are trainable. Official LVD-142M backbone weights are downloaded through the shared cache and hashed.

The published configuration is Adam learning rate 1e-6, batch 32, cross-entropy, 30 epochs, four loader workers, pinned memory and prefetch factor two. Section 5.1 explicitly reports the result after 30 epochs, so the last checkpoint is evaluated; no test-score checkpoint selection is performed. Full float32 is used. Seed 42, default Adam betas/epsilon/zero weight decay, bilinear resize and zero affine fill are routine reconstruction choices. The official XFORMERS_DISABLED option selects PyTorch scaled-dot-product attention to avoid framework-specific xFormers binary incompatibility, without changing attention mathematics or patching the backbone.

The container starts from the study GPU base image pinned to sha256:679f06c1abe55eda74213fe883e24db8802e8aeeac1b6d94619055b08029d2ea. Its resolved framework/package versions, driver, GPU and model structure are captured with every run. The Dockerfile pins upstream Git and does not install the obsolete full DINOv2 training requirements: official backbone inference/fine-tuning only needs PyTorch already supplied by the base.

## Exact data

Use [BDSL49 Mendeley V6](https://data.mendeley.com/datasets/k5yk4j8z8s/6), DOI 10.17632/k5yk4j8z8s.6, CC BY 4.0, Recognition_1.zip and Recognition_2.zip. The source [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10331282/) describes adult volunteers, hands-only collection and public research/academic/educational reuse. These terms permit this project-cloud processing of existing data. No new participants are enrolled and dataset images are not committed or republished. The tracker Kaggle lead contains the detection/full-frame material; it is not silently substituted for the original cropped recognition images.

The paper's total 14,745 is inconsistent with its own 11,774 train + 2,940 test. The authoritative release contains 14,714 images in each detection/recognition section, matching the intended split exactly. The original train/test membership is retained, and all 49 classes have 60 test images. The population manifest records archive hashes, every image hash and counts. Canonical data live at modal://datasets/bdsl49-v6-recognition, and training mounts datasets read-only. Hugging Face cache is mounted read-write at /cache/huggingface with both HF_HOME and HF_HUB_CACHE set. It is an optimization, not evidence storage; final artifacts live in a paper-specific results volume.

The official Mendeley link returned HTTP403 to programmatic clients. The ordinary public browser download worked, revealing the public storage redirect for Recognition_1. Recognition_2 was downloaded normally and uploaded through the workspace wrapper; its local and remote SHA256 matched before the task-specific local download was removed to free disk space. No access controls or click-through terms were bypassed. The failed direct acquisition is retained separately from the successful browser-source population.

Train preprocessing follows section 3.2: PIL RGB, resize224×224, horizontal flip p=.4, affine p=.4 with ±10° rotation, ±10% translation and scale .9–1.1; ColorJitter p=.4; then ImageNet mean/std. ColorJitter magnitudes are missing, so brightness/contrast/saturation .2 and hue .1 are declared guesses. Section 4 and Figure 4 explicitly specify test-set augmentation, so the primary evaluation applies the same transforms with a final fixed single-draw seed 314159, chosen before any scores. Figure 9 displays original-looking examples, which does not establish what the model saw. A separate deterministic resize/normalization evaluation is retained as diagnostic evidence. No protocol is selected by score closeness and no score-driven tuning is performed.

## Repeat commands

From the repository root:

```bash
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/tasnim-2026-dinov2-bdsl/scripts/modal_app.py --mode data
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/tasnim-2026-dinov2-bdsl/scripts/modal_app.py --mode preflight
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/tasnim-2026-dinov2-bdsl/scripts/modal_app.py --mode full
python3 .agents/skills/reproduce-paper/scripts/validate_reproduction.py papers/tasnim-2026-dinov2-bdsl
```

Both canonical volume handles explicitly require v2. The data entry point is idempotent after manifest creation. Both normal browser-observed public storage redirects are now pinned with archive checksums in data.sh, so a fresh population is automated. If those public redirects change, obtain the same archives through the ordinary public browser link, verify/upload them via the wrapper, then rerun data to extract and verify. Training copies hash-verified canonical images to ephemeral local storage for decode throughput. Outputs are idempotent when metrics exist; otherwise last.pt resumes model, optimizer, epoch and Python/NumPy/Torch/CUDA/loader RNG states. Evaluation is integrated and preserves logits, predictions, confusion matrix and metric calculations.

The representative preflight uses four real training images and two real test images per class, two epochs, a checkpoint roundtrip and an actual resumed optimizer step. Full training uses all official images for 30 epochs. Raw metrics, package freeze, model representation, hardware, history, checkpoint and predictions are retained on Modal with hashes in reproduction.json. No author contact occurred.

Independent review found no result-changing discrepancy. The retained execution uses official weights hash-verified during preflight and checks their final SHA256 against the predeclared artifact before accepting results. The committed entry point additionally asserts that checksum before loading weights, so a future cache mismatch fails immediately. The executed source remains retained separately by hash; the added assertion changes no model settings.

## Results and retained attempts

**Pipeline status:** complete

**Numerical agreement:** does_not_agree

**Preference level:** 3

| Table 5 target | Published | Reproduced | Difference |
|---|---:|---:|---:|
| Accuracy | 99.7% | 99.2177% | -0.4823 percentage points |
| Macro F1 | 99.7% | 99.2168% | -0.4832 percentage points |

The primary evaluation classified 2917 of 2940 images correctly after all 30 epochs, using the fixed augmented-test seed declared before training. Numerical agreement uses a ±0.05 percentage-point rounding bound, separate from completion; no settings were tuned toward the published score. The separate deterministic-test diagnostic is preserved in the raw metrics and has no role in selecting the primary result.

The full run used one A10G for 34.53 measured minutes (0.575 GPU-hours), excluding container startup. Peak allocated GPU memory was 3.128 GB. Estimated runtime cost is CHF 1.44; actual billed cost is unavailable. The prospective ceiling was four hours and CHF 10. Modal app `ap-RilRDcbPMrjvAWeXA1RURm`, call `fc-01M3VHTP5ANE30S6BSG6TEHT0P`, in the `repro-sign` main environment produced the native artifacts at `modal://tasnim-2026-dinov2-bdsl-results/full-seed42/`; reproduction.json links every metric and output checksum.

The initial direct Mendeley acquisition returned HTTP403 and was retained as failed; ordinary browser acquisition resolved it. The first GPU preflight was stopped before training because copying thousands of individual cold-volume files was slow. The corrected staging copies and verifies the two archives sequentially, extracts locally, then verifies every image hash. Its representative preflight completed in 40.60 seconds with checkpoint reload and optimizer resume verified; the full run then completed without a restart. Final backbone SHA256 is the same predeclared `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

No unresolved question requires human action. The dataset-total typo is resolved by the released split; unspecified ColorJitter strengths, defaults and seed remain disclosed reconstruction assumptions. Scientific review accepted the preserved Table 5 comparative advantage on 2026-10-03; the raw numerical differences remain unchanged.

Independent recomputation from the hash-verified raw predictions confirmed 2,940 unique test indices, 60 true examples per class, logits/argmax consistency, the confusion matrix, accuracy and macro F1. The deterministic-test diagnostic scored 99.4558% accuracy and 99.4562% macro F1; it remains secondary evidence and was never selected as the primary result.

## Conclusion assessment (2026-10-03)

The Table 5 comparative finding is reproduced and accepted as a valid in-scope reproduction. Our primary accuracy/F1 of **99.2177%/99.2168%** exceed the strongest listed CNN accuracy (**98.10%**, ensemble) and F1 (**98%**, quantized modified Xception), by **1.1177/1.2168 percentage points**. Other listed CNN accuracies are **93/87/91/86/91%** and F1 values **93/91/92%** (Table 5 and §5.3, PDF p.14). The exact **99.7%/99.7%** values were not matched.

Table 4 addresses a separate transformer comparison outside this assigned Table 5 conclusion. Substituting our primary accuracy into Table 4 places it below the reported ViT-B **99.8%**, ViT-L **99.5%**, and Swin-B **99.3%**, but above Swin-S **99.1%**. Thus the §5.2 claim of superiority over both Swin variants is not corroborated by this reconstruction. These CNN and transformer comparators were not rerun, so this is a comparison with published numbers, not a new controlled model ranking. Inference speed, the claimed accuracy/speed trade-off, fine-tuning ablations and cross-domain generalization were outside the executed targets. One seed cannot establish statistical equivalence or a stable ranking; the deterministic-test diagnostic remains secondary. No further human execution decision follows from these interpretation limits.

The existing independent review GPT-6/Codex agent performed this paper-to-result assessment and report update without new experiments. Exact model/build and application versions remain unavailable; its session attestation is recorded in the ledger.

On 2026-10-03, the GPT-6/Codex orchestrator verified the Table 5 comparisons and recorded the accepted scientific interpretation. This was later report maintenance, not a new experiment; all target values, execution attribution, artifact hashes and the exact numerical-agreement criterion are preserved.
