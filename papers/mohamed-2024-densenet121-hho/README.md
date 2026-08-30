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

Short answer on the main claim: no, I couldn't reproduce it. DenseNet121-HHO's 99.79% test accuracy came out as 95.11% here — a 4.68-point gap. It's not just below the paper's number, either: in my results it's below six of the other seven models — EfficientNet-B0 (99.39%), EfficientNet-B3 (99.14%), CNN (97.19%), DenseNet201 (96.99%), its own non-HHO DenseNet121 (96.39%), and ResNet50 (95.37%) all score higher; only CNN-HHO (91.05%) scores lower. The paper claims the opposite — that DenseNet121-HHO is the best of the eight — so this isn't just a reproduction gap on one number, it's a different ranking.

The two biggest reasons for the overall gap: the paper is ambiguous about how much of each pretrained backbone to freeze, and my first, simpler reading (freeze the whole thing) turned out to leave the pretrained models with far too few trainable parameters to compete — switching to partial unfreezing closed most of the gap (see "How much of each pretrained backbone to freeze"). And for the two HHO-tuned models, the paper gives no population size, iteration count, or search budget for Harris Hawks Optimization, so I had to pick my own — a reasonable but necessarily different search than whatever the authors ran, which is at least part of why CNN-HHO and DenseNet121-HHO still trail their non-HHO counterparts by more than the rest (see "The HHO search itself").

## Where I looked for code and data

I searched pretty broadly for existing code: the paper's own PDF and reference list have no code or data-availability statement, and I couldn't find anything on GitHub or the web under the paper's title, the authors' names, or the distinctive "DenseNet121-HHO" / "CNN-HHO" naming. There are other Arabic Sign Language repos out there (e.g. `Alkholy53/Arabic_sign_language`, `pavlyhalim/Arabic-Sign-Language`), but none of them implement this paper's method. So I wrote all eight models myself, from the paper's text and figures alone — preference level 3.

The dataset was easier: the paper uses ArASL2018 (54,049 images, 32 classes), and it was already sitting on our shared `datasets` volume at `arasl-database-grayscale`, mirrored from [`pain/ArASL_Database_Grayscale`](https://huggingface.co/datasets/pain/ArASL_Database_Grayscale) on Hugging Face (CC BY 4.0). 

## Results

All 32 numbers below came out of real training runs — nothing here is estimated or interpolated. Every model trained for exactly the epoch count Table II lists, on the same 64/16/20 stratified split (seed 42), and got evaluated once at the end, the same way the paper describes doing it (no picking the best checkpoint along the way). The pretrained models below use partial backbone unfreezing rather than freezing the whole backbone — see "How much of each pretrained backbone to freeze" for why.

| System | Metric | Paper (Table II) | Mine | Difference |
| --- | --- | ---: | ---: | ---: |
| CNN | Test Accuracy | 98.70 | 97.19 | -1.51 |
| CNN | Precision | 94.05 | 97.28 | +3.23 |
| CNN | Recall | 91.53 | 97.12 | +5.59 |
| CNN | F1 | 92.77 | 97.15 | +4.38 |
| CNN-HHO | Test Accuracy | 98.87 | 91.05 | -7.82 |
| CNN-HHO | Precision | 95.25 | 91.81 | -3.44 |
| CNN-HHO | Recall | 93.46 | 91.16 | -2.30 |
| CNN-HHO | F1 | 94.25 | 91.12 | -3.13 |
| EfficientNet-B0 | Test Accuracy | 99.13 | 99.39 | +0.26 |
| EfficientNet-B0 | Precision | 98.54 | 99.40 | +0.86 |
| EfficientNet-B0 | Recall | 98.00 | 99.39 | +1.39 |
| EfficientNet-B0 | F1 | 98.27 | 99.39 | +1.12 |
| EfficientNet-B3 | Test Accuracy | 99.17 | 99.14 | -0.03 |
| EfficientNet-B3 | Precision | 97.07 | 99.17 | +2.10 |
| EfficientNet-B3 | Recall | 96.67 | 99.17 | +2.50 |
| EfficientNet-B3 | F1 | 96.61 | 99.16 | +2.55 |
| ResNet50 | Test Accuracy | 99.16 | 95.37 | -3.79 |
| ResNet50 | Precision | 97.68 | 95.47 | -2.21 |
| ResNet50 | Recall | 96.46 | 95.38 | -1.08 |
| ResNet50 | F1 | 96.91 | 95.39 | -1.52 |
| DenseNet201 | Test Accuracy | 99.21 | 96.99 | -2.22 |
| DenseNet201 | Precision | 98.12 | 97.09 | -1.03 |
| DenseNet201 | Recall | 97.71 | 96.93 | -0.78 |
| DenseNet201 | F1 | 97.77 | 96.98 | -0.79 |
| DenseNet121 | Test Accuracy | 99.79 | 96.39 | -3.40 |
| DenseNet121 | Precision | 98.51 | 96.54 | -1.97 |
| DenseNet121 | Recall | 99.97 | 96.38 | -3.59 |
| DenseNet121 | F1 | 98.57 | 96.38 | -2.19 |
| DenseNet121-HHO | Test Accuracy | 99.79 | 95.11 | -4.68 |
| DenseNet121-HHO | Precision | 99.39 | 95.59 | -3.80 |
| DenseNet121-HHO | Recall | 99.95 | 94.96 | -4.99 |
| DenseNet121-HHO | F1 | 99.22 | 95.09 | -4.13 |

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

The `--model` flag takes one of `cnn`, `cnn-hho`, `efficientnet-b0`, `efficientnet-b3`, `resnet50`, `densenet201`, `densenet121`, `densenet121-hho`.

## The dataset

It's the Hugging Face mirror `pain/ArASL_Database_Grayscale` (revision `114709884276379a01e0722d71cd590c8ad3a05d`), a single `train` split stored as one parquet file (`data/train-00000-of-00001-aa6a48ea2f282316.parquet`, SHA-256 `7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3`), CC BY 4.0, sitting on the shared `datasets` volume at `arasl-database-grayscale`. I split it myself 64/16/20 (stratified, seed 42) since the paper doesn't publish its own split.

## Environment

I used `tensorflow/tensorflow:2.17.0-gpu` rather than the repo's usual PyTorch base image, since this paper's whole stack is Keras — DenseNet/ResNet/EfficientNet from `tf.keras.applications`, plus the usual `ImageDataGenerator`-style augmentation. On top of that: `pandas`, `pyarrow`, `scikit-learn`, `pillow`. Each of the 8 models trained on its own single A10G GPU on Modal.

## Hyperparameters

Each model's final hyperparameters (some picked by HHO, not by me), and how many of each pretrained backbone's own layers were left trainable instead of frozen (see "How much of each pretrained backbone to freeze" below for why):

| Model | Batch | Dropout | Learning rate | Backbone layers unfrozen |
| --- | ---: | ---: | ---: | --- |
| CNN | 32 | 0.500 | 1e-4 | n/a (no backbone) |
| CNN-HHO | 29 | 0.200 | 1e-4 (found by HHO) | n/a (no backbone) |
| EfficientNet-B0 | 32 | 0.600 | 1e-3 (Adamax) | all 234 |
| EfficientNet-B3 | 32 | 0.600 | 1e-3 (Adamax) | last 35 of 381 |
| ResNet50 | 32 | 0.500 | Adamax default | last 10 of 174 |
| DenseNet201 | 32 | 0.800 | Adamax default | last 125 of 704 |
| DenseNet121 | 32 | 0.500 | Adam default (1e-3) | last 200 of 424 |
| DenseNet121-HHO | 34 | 0.200 | 1.97e-4 (found by HHO) | last 200 of 424 |

## How much of each pretrained backbone to freeze

The paper's wording here is genuinely ambiguous: "the early layers of the network were frozen... focusing the training process on the remaining layers for fine-tuning" (repeated almost word-for-word for EfficientNet, ResNet50, and DenseNet201/121). That sentence supports two quite different readings — freeze the entire pretrained backbone and train only the new head on top, or freeze just the first few blocks and fine-tune the rest.

I tried both. Freezing the whole backbone is the simpler and more common reading of that kind of sentence, so it's what I built first. It trains only 262K–543K parameters per pretrained model, against the from-scratch CNN's 4.5M, and it showed in the results: the CNN beat every pretrained model, and the deepest one, DenseNet201, did worst of all — backwards from what you'd normally expect.

So I went with the second reading instead: partial unfreezing. To pick how many layers without just tuning until some model's accuracy looked good, I built each backbone once, counted trainable parameters as a function of how many trailing layers stay unfrozen, and picked the smallest count that gets close to the from-scratch CNN's ~4.5M trainable parameters — a stopping rule based on parameter count, not on any accuracy number. That landed on: ResNet50, last 10 of 174 layers; EfficientNet-B3, last 35 of 381; DenseNet201, last 125 of 704; DenseNet121 (both variants), last 200 of 424; EfficientNet-B0 can't reach 4.5M even fully unfrozen, so all 234 of its layers are trainable. These counts are hardcoded in `train.py`'s `make_model_specs()`, not derived at runtime.

The results table above reflects this second reading, and the effect was large: EfficientNet-B0 and EfficientNet-B3 now land within a quarter of a point of the paper, and everything else closed most of its gap (DenseNet201 from 28 points behind to 2, DenseNet121 from 14 to 3.4, ResNet50 from 18 to 3.8).

## Guesses I had to make

The paper leaves out a lot of implementation detail, so here's everywhere I had to fill in a gap myself, and what I think the consequence is:

- **No horizontal flip.** Algorithm 1's actual augmentation list is rotation, zoom, shift, and brightness — no flip. Fig. 8's illustrative box for the DenseNet121-HHO pipeline does mention "Horizontal Flip" instead of rotation, which is a small inconsistency between the two. I went with Algorithm 1 (the one place the paper actually spells out the augmentation procedure) and left flipping out everywhere.
- **How much of each pretrained backbone is frozen.** Covered above under "How much of each pretrained backbone to freeze" — I unfroze a hardcoded number of trailing layers per backbone, chosen to match the CNN's trainable-parameter count rather than to match any accuracy number. It's still a guess in the sense that "match the CNN's parameter count" is my own stopping rule, not something the paper states, and the paper gives no layer count for "early" either way.
- **Validation is augmented, not just training.** Algorithm 1 says the augmented generator covers training *and* validation, and only the test generator skips augmentation — so `make_dataset` augments both training and validation, and only test gets plain normalization. This also means the fitness signal the two HHO searches optimize against is computed on augmented validation data.
- **DenseNet121's final activation.** The paper's text says the last layer uses sigmoid for 32-way classification, which doesn't really make sense for a single-label, mutually-exclusive classification problem — every other model in the paper uses softmax for the same task. I used softmax and treated the sigmoid mention as a slip in the writing.
- **Learning rate**, for the plain CNN, ResNet50, DenseNet121, and DenseNet201 — the paper names an optimizer (Adam or Adamax) for each but never gives a rate. I used the framework default; for the CNN, I used 1e-4 for the reasons above.
- **The ResNet50 head's dropout rate** — the paper's Figure 5 just says "Flatten → Dropout → Dense" with no number. I used 0.5, a plain default, distinct from the two dropout rates the paper does give for other models (0.6 for EfficientNet, 0.8 for DenseNet201).
- **How precision/recall/F1 are averaged across the 32 classes.** The paper never says. I used macro averaging (unweighted mean across classes), which is the standard choice when it isn't specified — and it's also consistent with something I noticed in the paper's own numbers: their F1 is sometimes lower than both their precision and recall, which can't happen with micro-averaging (where all three collapse toward the same value as accuracy) but is completely normal under macro or per-class averaging on an imbalanced label set like this one.
- **The actual train/val/test split.** The paper says 64/16/20, stratified, but doesn't publish the split itself or a seed. I used seed 42.
- **The HHO search itself.** The paper says HHO tunes hyperparameters "to maximize validation accuracy" but gives no population size, number of iterations, or how expensive each candidate evaluation should be. Evaluating every candidate with a full training run would cost far more than the 5-epoch final run the paper reports, so I used a population of 6, 4 iterations, and judged each candidate by training it for 3 epochs on a 25% slice of the training data — a fairly standard cheap-proxy setup for this kind of search. It's worth noting this search doesn't always land somewhere good: even with the partial-unfreezing fix, DenseNet121-HHO's search still settled on a configuration that does slightly *worse* than the plain DenseNet121 default (95.11% vs. 96.39%), which I'm reporting as-is rather than re-running the search until it beats the baseline.