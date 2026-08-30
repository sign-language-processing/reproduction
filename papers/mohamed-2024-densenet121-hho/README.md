# DenseNet121 with Harris Hawks Optimization (ArSL recognition) reproduction

**Paper ID:** `mohamed-2024-densenet121-hho`

**Citation:** S. N. Mohamed, H. Hussein, and M. S. Elgamel, "DenseNet121 with Harris Hawks Optimization: A Novel Deep Learning Approach for Arabic Sign Language Recognition," in *2024 International Conference on Computer and Applications (ICCA)*, 2024. DOI: [10.1109/ICCA62237.2024.10928112](https://ieeexplore.ieee.org/document/10928112/)

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/10928112/) (PDF SHA-256 `b0c54b9bdb123ac278d3317418c6fcb888ab99a1b7f8b312aabc2629d008de3c`) · **Code:** none — I couldn't find any (see below)

**Preference level:** 3

I ended up writing this from scratch. There's no code released with the paper.

**Status:** `complete`

**Numerical agreement:** `not_fully_reproduced`

**Attempt date:** 2026-08-30

## Summary

The paper trains and compares eight models on Arabic Sign Language alphabet recognition: a plain CNN, that same CNN tuned with Harris Hawks Optimization (HHO), EfficientNet-B0, EfficientNet-B3, ResNet50, DenseNet201, DenseNet121, and DenseNet121 tuned with HHO. All eight are trained and evaluated on the same ArASL2018 dataset with the same 64/16/20 train/val/test split, and Table II reports all of them side by side. The abstract's "an impressive 99.79 percent" is DenseNet121-HHO's row in that table.

So the target list is straightforward: for each of the 8 models, the four metrics the abstract says it reports (test accuracy, precision, recall, F1) — 32 numbers in total, all read off Table II. I left out the training-accuracy/loss and validation-accuracy/loss columns since those are training curves, not results, though they're still saved per-epoch alongside everything else.

## Where I looked for code and data

I searched pretty broadly for existing code: the paper's own PDF and reference list have no code or data-availability statement, and I couldn't find anything on GitHub or the web under the paper's title, the authors' names, or the distinctive "DenseNet121-HHO" / "CNN-HHO" naming. There are other Arabic Sign Language repos out there (e.g. `Alkholy53/Arabic_sign_language`, `pavlyhalim/Arabic-Sign-Language`), but none of them implement this paper's method. So I wrote all eight models myself, from the paper's text and figures alone — preference level 3.

The dataset was easier: the paper uses ArASL2018 (54,049 images, 32 classes), and it was already sitting on our shared `datasets` volume at `arasl-database-grayscale`, mirrored from [`pain/ArASL_Database_Grayscale`](https://huggingface.co/datasets/pain/ArASL_Database_Grayscale) on Hugging Face (CC BY 4.0). 

## Results

All 32 numbers below came out of real training runs — nothing here is estimated or interpolated. Every model trained for exactly the epoch count Table II lists, on the same 64/16/20 stratified split (seed 42), and got evaluated once at the end, the same way the paper describes doing it (no picking the best checkpoint along the way).

| System | Metric | Paper (Table II) | Mine | Difference |
| --- | --- | ---: | ---: | ---: |
| CNN | Test Accuracy | 98.70 | 96.28 | -2.42 |
| CNN | Precision | 94.05 | 96.57 | +2.52 |
| CNN | Recall | 91.53 | 96.28 | +4.75 |
| CNN | F1 | 92.77 | 96.36 | +3.59 |
| CNN-HHO | Test Accuracy | 98.87 | 87.32 | -11.55 |
| CNN-HHO | Precision | 95.25 | 88.40 | -6.85 |
| CNN-HHO | Recall | 93.46 | 87.49 | -5.97 |
| CNN-HHO | F1 | 94.25 | 87.38 | -6.87 |
| EfficientNet-B0 | Test Accuracy | 99.13 | 91.09 | -8.04 |
| EfficientNet-B0 | Precision | 98.54 | 91.54 | -7.00 |
| EfficientNet-B0 | Recall | 98.00 | 90.94 | -7.06 |
| EfficientNet-B0 | F1 | 98.27 | 90.97 | -7.30 |
| EfficientNet-B3 | Test Accuracy | 99.17 | 86.85 | -12.32 |
| EfficientNet-B3 | Precision | 97.07 | 87.67 | -9.40 |
| EfficientNet-B3 | Recall | 96.67 | 86.91 | -9.76 |
| EfficientNet-B3 | F1 | 96.61 | 87.10 | -9.51 |
| ResNet50 | Test Accuracy | 99.16 | 81.09 | -18.07 |
| ResNet50 | Precision | 97.68 | 83.34 | -14.34 |
| ResNet50 | Recall | 96.46 | 81.27 | -15.19 |
| ResNet50 | F1 | 96.91 | 80.86 | -16.05 |
| DenseNet201 | Test Accuracy | 99.21 | 70.72 | -28.49 |
| DenseNet201 | Precision | 98.12 | 75.44 | -22.68 |
| DenseNet201 | Recall | 97.71 | 70.57 | -27.14 |
| DenseNet201 | F1 | 97.77 | 70.07 | -27.70 |
| DenseNet121 | Test Accuracy | 99.79 | 85.74 | -14.05 |
| DenseNet121 | Precision | 98.51 | 86.47 | -12.04 |
| DenseNet121 | Recall | 99.97 | 85.54 | -14.43 |
| DenseNet121 | F1 | 98.57 | 85.67 | -12.90 |
| DenseNet121-HHO | Test Accuracy | 99.79 | 81.43 | -18.36 |
| DenseNet121-HHO | Precision | 99.39 | 82.64 | -16.75 |
| DenseNet121-HHO | Recall | 99.95 | 81.33 | -18.62 |
| DenseNet121-HHO | F1 | 99.22 | 81.20 | -18.02 |

So: every model trained fine and landed well above chance, but every single number comes in below the paper — most by a lot. I don't think this is a scoreboard-vs-me situation where one variant matches and the rest don't; it's a consistent gap across the board, which makes me suspect something structural (a modeling choice I got wrong, or the paper's numbers being optimistic) rather than a bug in any one model. More on that below.

## How to run it

```bash
./setup.sh
# sanity-check the dataset volume is what I think it is:
.agents/skills/reproduce-paper/scripts/check_modal_dataset.sh arasl-database-grayscale

# quick smoke test for any one of the 8 models:
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
  papers/mohamed-2024-densenet121-hho/modal_app.py --model densenet121 --preflight-only

# the real thing (one Modal GPU call per model):
.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run -d \
  papers/mohamed-2024-densenet121-hho/modal_app.py --model densenet121
```

The `--model` flag takes one of `cnn`, `cnn-hho`, `efficientnet-b0`, `efficientnet-b3`, `resnet50`, `densenet201`, `densenet121`, `densenet121-hho`. Each one writes its results under its own folder on the `mohamed-2024-densenet121-hho-results` volume, and a full run just returns the existing `run.json` if one's already there for that model — so if you want to force a clean re-run, delete that model's folder on the volume first.

## The dataset

It's the Hugging Face mirror `pain/ArASL_Database_Grayscale` (revision `114709884276379a01e0722d71cd590c8ad3a05d`), a single `train` split stored as one parquet file (`data/train-00000-of-00001-aa6a48ea2f282316.parquet`, SHA-256 `7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3`), CC BY 4.0, sitting on the shared `datasets` volume at `arasl-database-grayscale`. I split it myself 64/16/20 (stratified, seed 42) since the paper doesn't publish its own split.

## Environment

I used `tensorflow/tensorflow:2.17.0-gpu` rather than the repo's usual PyTorch base image, since this paper's whole stack is Keras — DenseNet/ResNet/EfficientNet from `tf.keras.applications`, plus the usual `ImageDataGenerator`-style augmentation. On top of that: `pandas`, `pyarrow`, `scikit-learn`, `pillow`. Each of the 8 models trained on its own single A10G GPU on Modal.

## Runs

All 8 runs finished cleanly (`succeeded` / `completed`), each on its own A10G on Modal, profile `repro-sign`.

| Run | Modal app | Started → finished (UTC) | Epochs | GPU-hours | Evidence |
| --- | --- | --- | --- | --- | --- |
| `cnn-full` | `ap-K6CxkuW4iMNIYNwYaM9LJT` | 06:23:07 → 06:27:07 | 30 | 0.07 | `run.json` sha256 `46c0658cc098cd4f7039e1c676b02a303300ccea01057f0d00615ac0b78937b0` |
| `cnn-hho-full` | `ap-ajwUKiKnHguHYyQlNjTRwK` | 06:03:49 → 06:22:48 | 5 (+ HHO search) | 0.32 | `run.json` sha256 `3445d9a5746eb1d992bf3f349bf845389e8b68dd2735b1afc079e88a194221df`; `hho_search.json` sha256 `22bd09adce746b506c0a18fe5d7e854629b4cb3e68da1d1f8ec588fe6b05611c` |
| `efficientnet-b0-full` | `ap-Amnqe2zxHkEtEfxFZz9iig` | 06:23:09 → 06:48:32 | 70 | 0.42 | `run.json` sha256 `99237b4b969a11871ae18e1e36b98944b0deb59ca680d9e20e77104fc7ef8fae` |
| `efficientnet-b3-full` | `ap-2aQAb7ptlpfN1NJsgmmxiC` | 06:23:10 → 07:18:17 | 84 | 0.92 | `run.json` sha256 `e9e2cb32b74dfd3a8b51f61de4d9263453548b8b7e1e89420d195629dbdc4435` |
| `resnet50-full` | `ap-L0OoR0UFkFI2rzkymBKcKc` | 06:23:11 → 06:33:12 | 60 | 0.17 | `run.json` sha256 `a8819b3650de2e5c65729e008c53ce1b947673ff996e25e1997b33f7f59bacf9` |
| `densenet201-full` | `ap-bcng5McXkGeEcCOiznJR45` | 06:23:13 → 07:07:43 | 150 | 0.74 | `run.json` sha256 `0aebe3834b9bfdeaf80fbe111707597b90e8d0fbe5e669f6aa52ac4e6bb7cb5f` |
| `densenet121-full` | `ap-bsPDq91XaYCKr1C1naRuRl` | 06:23:14 → 06:27:52 | 12 | 0.08 | `run.json` sha256 `fc3fa2239089f932b8d54bc3a34a17362038cf7ae6114425441d46027a1ae452` |
| `densenet121-hho-full` | `ap-T1STpdAZ97XFGfgadV5ufV` | 06:23:15 → 07:12:27 | 5 (+ HHO search) | 0.82 | `run.json` sha256 `efe77a90a7b3035fc31926d8f8c0f2ad949e4b066ed613485b255bb5d023a152`; `hho_search.json` sha256 `a15065d1da97f4c38daac2641f701ebaca01bc3c4d10c755b24c3a05015c4136` |

Each model's final hyperparameters (some of these were picked by HHO, not by me):

| Model | Batch | Dropout | Learning rate |
| --- | ---: | ---: | ---: |
| CNN | 32 | 0.500 | 1e-4 |
| CNN-HHO | 18 | 0.200 | 1e-4 (found by HHO) |
| EfficientNet-B0 | 32 | 0.600 | 1e-3 (Adamax) |
| EfficientNet-B3 | 32 | 0.600 | 1e-3 (Adamax) |
| ResNet50 | 32 | 0.500 | Adamax default |
| DenseNet201 | 32 | 0.800 | Adamax default |
| DenseNet121 | 32 | 0.500 | Adam default (1e-3) |
| DenseNet121-HHO | 55 | 0.377 | 1.94e-3 (found by HHO) |

## Guesses I had to make, and what I think is actually going on

The paper leaves out a lot of implementation detail, so here's everywhere I had to fill in a gap myself, and what I think the consequence is:

- **How much of each pretrained backbone is frozen.** The paper just says "the early layers... were frozen... focusing training on the remaining layers," for all four pretrained backbones. I read that as "freeze the whole pretrained backbone, train only the new head on top" — the standard reading of that kind of sentence, and it's what all four model descriptions in the paper say almost word-for-word. But it could also mean "freeze only the first few blocks and fine-tune the rest," which is a very different (and much more expensive) training recipe. I didn't try that second reading, partly because the paper gives no indication of how many layers "early" means, and partly because I didn't want to go hunting for the unfreezing depth that happens to close the gap to 99% — that's exactly the kind of tuning-toward-the-score I'm supposed to avoid. But I do think this is the single most likely explanation for the accuracy gap (see below), and it's the one place I'd point a reviewer who wants to dig further.
- **DenseNet121's final activation.** The paper's text says the last layer uses sigmoid for 32-way classification, which doesn't really make sense for a single-label, mutually-exclusive classification problem — every other model in the paper uses softmax for the same task. I used softmax and treated the sigmoid mention as a slip in the writing.
- **Learning rate**, for the plain CNN, ResNet50, DenseNet121, and DenseNet201 — the paper names an optimizer (Adam or Adamax) for each but never gives a rate. I used the framework default; for the CNN, I used 1e-4 for the reasons above.
- **The ResNet50 head's dropout rate** — the paper's Figure 5 just says "Flatten → Dropout → Dense" with no number. I used 0.5, a plain default, distinct from the two dropout rates the paper does give for other models (0.6 for EfficientNet, 0.8 for DenseNet201).
- **How precision/recall/F1 are averaged across the 32 classes.** The paper never says. I used macro averaging (unweighted mean across classes), which is the standard choice when it isn't specified — and it's also consistent with something I noticed in the paper's own numbers: their F1 is sometimes lower than both their precision and recall, which can't happen with micro-averaging (where all three collapse toward the same value as accuracy) but is completely normal under macro or per-class averaging on an imbalanced label set like this one.
- **The actual train/val/test split.** The paper says 64/16/20, stratified, but doesn't publish the split itself or a seed. I used seed 42.
- **The HHO search itself.** The paper says HHO tunes hyperparameters "to maximize validation accuracy" but gives no population size, number of iterations, or how expensive each candidate evaluation should be. Evaluating every candidate with a full training run would cost far more than the 5-epoch final run the paper reports, so I used a population of 6, 4 iterations, and judged each candidate by training it for 3 epochs on a 25% slice of the training data — a fairly standard cheap-proxy setup for this kind of search. It's worth noting this search doesn't always land somewhere good: DenseNet121-HHO's search settled on a configuration that actually does *worse* than the plain DenseNet121 (81.4% vs. 85.7%), which I'm reporting as-is rather than re-running the search until it beats the baseline.

**On the gap itself:** after both bug fixes above, every model trains normally — all comfortably above chance, in a range that's plausible for a frozen pretrained backbone on a 32-class hand-shape task at 64×64 resolution — but every one still lands well below its Table II number, by anywhere from 2 to nearly 30 points. What stands out to me is the pattern, not just the size of the gap: the from-scratch CNN (96.3%) outperforms every pretrained model I tried, and the model trained the longest with the most capacity, DenseNet201 (150 epochs), does the *worst* of the eight (70.7%). That's backwards from what you'd normally expect, and it points pretty strongly at the frozen-backbone choice above being the main bottleneck — a fully frozen ImageNet backbone just isn't going to adapt as well to 64×64 hand-sign images as a network that's allowed to fine-tune its later layers.

## What went wrong along the way

- The very first `cnn` and `cnn-hho` runs sat at chance accuracy for the reason described above (learning rate). I found this from CNN-HHO's own search consistently landing on 1e-4, then confirmed it with a short 8-epoch test.
- The first `efficientnet-b0` and `efficientnet-b3` runs I killed manually partway through once "chance accuracy, no matter what I train" started looking like a pattern rather than a fluke. My first attempted fix — passing `include_preprocessing=False` to the Keras EfficientNet constructor — actually crashed outright (`TypeError: got an unexpected keyword argument 'include_preprocessing'`, not supported on TF 2.17), so I switched to manually undoing the [0,1] scaling and applying the correct `preprocess_input`, which is the approach that ended up working for all four pretrained backbones.
- The first `resnet50` run didn't crash or get killed — it ran to completion and quietly produced a bad number (33.7%), which took a bit longer to notice as a bug rather than "well, maybe ResNet50 just isn't a great fit here."
- After fixing the code, my first attempt to re-run everything backfired in a different way: the training script treats an existing `run.json` as "already done," so relaunching `cnn`, `resnet50`, and `densenet121` just handed back their old, broken results without re-training at all. Worse, relaunching `efficientnet-b0`, `efficientnet-b3`, `densenet201`, and `densenet121-hho` silently *resumed* from the old checkpoints — the ones trained with the wrong preprocessing — instead of starting fresh. I caught this from a Keras warning about mismatched optimizer state on load, plus the numbers coming back identical to before. Fix was just to delete those models' folders from the results volume and relaunch clean, which is what produced the numbers reported above. CNN-HHO never had this problem — its first run was already the correct one.
- Along the way there were also a handful of small throwaway smoke-test and quick-test runs I used to check each fix cheaply (a few epochs on real data) before committing to the full paper-length runs. Those aren't recorded as retained runs since they didn't produce anything final — they were just there to catch bugs early.

## Contacting the authors

I didn't. There's no code to ask for, and I didn't reach out about the Table I/Table II inconsistencies before finishing this attempt — that's a reasonable next step for a human reviewer if they want to pursue it, along with the identical DenseNet121/DenseNet121-HHO number.
