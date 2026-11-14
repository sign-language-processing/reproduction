# DenseNet121 with Harris Hawks Optimization (ArSL recognition) reproduction

**Paper ID:** `e13b83983c37b44c5b555743fffd4d6cec545e4f`

**Citation:** S. N. Mohamed, H. Hussein, and M. S. Elgamel, "DenseNet121 with Harris Hawks Optimization: A Novel Deep Learning Approach for Arabic Sign Language Recognition," in *2024 International Conference on Computer and Applications (ICCA)*, 2024. DOI: [10.1109/ICCA62237.2024.10928112](https://ieeexplore.ieee.org/document/10928112/)

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/10928112/) (PDF SHA-256 `b0c54b9bdb123ac278d3317418c6fcb888ab99a1b7f8b312aabc2629d008de3c`) · **Code:** none — I couldn't find any (see below)

**Preference level:** 3

I ended up writing this from scratch. There's no code released with the paper.

**Pipeline status:** `complete`

**Numerical agreement:** `does_not_agree` — all 32 Table II targets produced comparable values; absolute differences from the published numbers run from 0.0303 to 7.8154 percentage points.

**Attempt date:** 2026-08-30

## Reproduction agents

| Agent ID | Model and version | Agent application | Contribution | Attribution evidence / unknowns |
| --- | --- | --- | --- | --- |
| `original-reproduction-agent` | unknown | unknown | Wrote the reimplementation and ran all eight retained runs. | Audited 2026-11-06: none of the three commits touching this directory carries a model or agent-application trailer — [`aaa7d1a`](https://github.com/sign-language-processing/reproduction/commit/aaa7d1a), `a3cb787`, `d8e6fcd` — and [PR #39](https://github.com/sign-language-processing/reproduction/pull/39) has an empty body with two comments and one review that only discuss merge permissions. |

I can't say which model or application did this work — the commits and the PR don't record it, and
a human account or branch name doesn't establish it. So both fields are null and all eight runs
carry `agent_ids: []` with an `agent_attribution_note`.

## Summary

The paper trains and compares eight models on Arabic Sign Language alphabet recognition: a plain CNN, that same CNN tuned with Harris Hawks Optimization (HHO), EfficientNet-B0, EfficientNet-B3, ResNet50, DenseNet201, DenseNet121, and DenseNet121 tuned with HHO. All eight are trained and evaluated on the same ArASL2018 dataset with the same 64/16/20 train/val/test split, and Table II reports all of them side by side. The abstract's "an impressive 99.79 percent" is DenseNet121-HHO's row in that table.

Short answer on the main claim: no, I couldn't reproduce it. DenseNet121-HHO's 99.79% test accuracy came out as 95.11% here — a 4.68-point gap. It's not just below the paper's number, either: in my results it's below six of the other seven models — EfficientNet-B0 (99.39%), EfficientNet-B3 (99.14%), CNN (97.19%), DenseNet201 (96.99%), its own non-HHO DenseNet121 (96.39%), and ResNet50 (95.37%) all score higher; only CNN-HHO (91.05%) scores lower. The paper claims the opposite — that DenseNet121-HHO is the best of the eight — so this isn't just a reproduction gap on one number, it's a different ranking.

The two biggest reasons for the overall gap: the paper is ambiguous about how much of each pretrained backbone to freeze, and my first, simpler reading (freeze the whole thing) turned out to leave the pretrained models with far too few trainable parameters to compete — switching to partial unfreezing closed most of the gap (see "How much of each pretrained backbone to freeze"). And for the two HHO-tuned models, the paper gives no population size, iteration count, or search budget for Harris Hawks Optimization, so I had to pick my own — a reasonable but necessarily different search than whatever the authors ran, which is at least part of why CNN-HHO and DenseNet121-HHO still trail their non-HHO counterparts by more than the rest (see `hho-search` under "Guesses and deviations").

## Where I looked for code and data

I searched pretty broadly for existing code: the paper's own PDF and reference list have no code or data-availability statement, and I couldn't find anything on GitHub or the web under the paper's title, the authors' names, or the distinctive "DenseNet121-HHO" / "CNN-HHO" naming. There are other Arabic Sign Language repos out there (e.g. `Alkholy53/Arabic_sign_language`, `pavlyhalim/Arabic-Sign-Language`), but none of them implement this paper's method. So I wrote all eight models myself, from the paper's text and figures alone — preference level 3.

The dataset was easier: the paper uses ArASL2018 (54,049 images, 32 classes), and it was already sitting on our shared `datasets` volume at `arasl-database-grayscale`, mirrored from [`pain/ArASL_Database_Grayscale`](https://huggingface.co/datasets/pain/ArASL_Database_Grayscale) on Hugging Face (CC BY 4.0). 

## Results

All 32 numbers below came out of real training runs — nothing here is estimated or interpolated. Every model trained for exactly the epoch count Table II lists, on the same 64/16/20 stratified split (seed 42), and got evaluated once at the end (no picking the best checkpoint along the way). The paper says Table II holds each model's "best test results" without saying best over what; see `checkpoint-selection`. The pretrained models below use partial backbone unfreezing rather than freezing the whole backbone — see "How much of each pretrained backbone to freeze" for why.

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

The paper's wording here is genuinely ambiguous: "the early layers of the network were frozen... focusing the training process on the remaining layers for fine-tuning" (said of DenseNet121; ResNet50's description says "with the early layers frozen", EfficientNet-B0/B3 are described only as "fine-tuned", and DenseNet201's description does not mention freezing). That sentence supports two quite different readings — freeze the entire pretrained backbone and train only the new head on top, or freeze just the first few blocks and fine-tune the rest.

I tried both. Freezing the whole backbone is the simpler and more common reading of that kind of sentence, so it's what I built first. It trains only 262K–543K parameters per pretrained model, against the from-scratch CNN's 4.5M, and it showed in the results: the CNN beat every pretrained model, and the deepest one, DenseNet201, did worst of all — backwards from what you'd normally expect.

So I went with the second reading instead: partial unfreezing. To pick how many layers without just tuning until some model's accuracy looked good, I built each backbone once, counted trainable parameters as a function of how many trailing layers stay unfrozen, and picked the smallest count that gets close to the from-scratch CNN's ~4.5M trainable parameters — a stopping rule based on parameter count, not on any accuracy number. That landed on: ResNet50, last 10 of 174 layers; EfficientNet-B3, last 35 of 381; DenseNet201, last 125 of 704; DenseNet121 (both variants), last 200 of 424; EfficientNet-B0 can't reach 4.5M even fully unfrozen, so all 234 of its layers are trainable. These counts are hardcoded in `train.py`'s `make_model_specs()`, not derived at runtime.

The results table above reflects this second reading, and the effect was large: EfficientNet-B0 and EfficientNet-B3 now land within a quarter of a point of the paper, and everything else closed most of its gap (DenseNet201 from 28 points behind to 2, DenseNet121 from 14 to 3.4, ResNet50 from 18 to 3.8).

## Guesses and deviations

The paper leaves out a lot of implementation detail, so here's everywhere I had to fill in a gap myself, and what I think the consequence is:

| ID | Kind | Description | Paper says | This attempt used | Basis | Effect | Scope | Status | Evidence | Related IDs | Affects | Dataset IDs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `horizontal-flip` | `paper_internal_conflict` | Whether augmentation includes a horizontal flip. | Algorithm 1's augmentation list is rotation, zoom, shift, and brightness, with no flip; Fig. 8's "Image processing" box lists Rescaling, Zooming, Shifting, Brightness Adjustment, and Horizontal Flip | Algorithm 1's list; no flipping anywhere | Algorithm 1 is the one place the paper actually spells out the augmentation procedure |  | All eight models |  |  |  |  |  |
| `backbone-freezing` | `guess` | How much of each pretrained backbone is frozen. | DenseNet121 (§III.B.5): "the early layers of the network were frozen... focusing the training process on the remaining layers for fine-tuning"; ResNet50 (§III.B.3): "with the early layers frozen to preserve pre-learned features". EfficientNet-B0/B3 are only "fine-tuned" (§III.B.2), and DenseNet201's description (§III.B.4) does not mention freezing. No layer count for "early" | Partial unfreezing: ResNet50 last 10 of 174 layers; EfficientNet-B3 last 35 of 381; DenseNet201 last 125 of 704; DenseNet121 (both variants) last 200 of 424; EfficientNet-B0 all 234 | The smallest trailing-layer count whose trainable-parameter count gets close to the from-scratch CNN's ~4.5M; a stopping rule based on parameter count, not on any accuracy number. "Match the CNN's parameter count" is this attempt's own rule, not something the paper states | The whole-backbone reading was built first. It trained only 262K-543K parameters per pretrained model and the CNN beat every pretrained model. Partial unfreezing closed most of the gap (DenseNet201 from 28 points behind to 2, DenseNet121 from 14 to 3.4, ResNet50 from 18 to 3.8) | EfficientNet-B0, EfficientNet-B3, ResNet50, DenseNet201, DenseNet121, DenseNet121-HHO |  | train.py make_model_specs() |  |  |  |
| `validation-augmentation` | `guess` | Whether validation data is augmented. | Algorithm 1 says the augmented generator covers training and validation; only the test generator skips augmentation | Training and validation augmented; test gets plain normalization |  | The fitness signal the two HHO searches optimize against is computed on augmented validation data | All eight models |  |  |  |  |  |
| `rgb-channels` | `guess` | How grayscale images become the RGB input the paper describes. | "All images were resized to 64 × 64 pixels and retained in RGB format" (§III.A); Figs. 2, 6, and 7 show 64x64x3 input | Every image converted to grayscale and resized to 64x64 (648 of 54,049 are not native 64x64 grayscale), then the single channel replicated into 3 channels | The native ArASL2018 images are 64x64 grayscale |  | All eight models |  | train.py comments at lines 26 and 63 |  |  | `arasl2018-grayscale` |
| `backbone-input-preprocessing` | `paper_deviation` | Input scaling for the pretrained backbones. | Algorithm 1 normalizes images to [0, 1] ahead of every model | The [0, 1] scaling is undone and each backbone's own `preprocess_input` applied (mean-subtracted BGR for ResNet50, x/127.5-1 for DenseNet, EfficientNet's internal Rescaling+Normalization) | Each tf.keras.applications backbone was pretrained with its own preprocessing on raw [0, 255] input | Without it, ResNet50 reached ~34% test accuracy after 60 epochs and EfficientNet never escaped chance accuracy | EfficientNet-B0, EfficientNet-B3, ResNet50, DenseNet201, DenseNet121, DenseNet121-HHO |  | train.py resizing_backbone() comment |  |  |  |
| `efficientnet-input-size` | `guess` | How 64x64 images become EfficientNet's 224x224 input. | Fig. 4: EfficientNet-B0/B3 take a 224x224x3 input; the shared preprocessing produces 64x64x3 | Upsize with a Keras Resizing layer to 224x224 |  |  | EfficientNet-B0, EfficientNet-B3 |  | train.py resizing_backbone() |  |  |  |
| `cnn-architecture` | `paper_deviation` | The CNN's convolutional stack. | Fig. 2: Conv2D(32), Conv2D(32), Conv2D(64), Conv2D(128), with feature maps 64x64x32, 32x32x32, 16x16x64, 14x14x128, 7x7x128 and a Flatten of 6272 before Dense(512) | Three blocks of two same-padded 3x3 convolutions (32, 64, 128 filters), each followed by MaxPooling and BatchNormalization; Flatten of 8192 before Dense(512) |  | About 4.5M trainable parameters, against about 3.3M for Fig. 2's shapes (the Flatten-to-Dense(512) layer alone is 4.2M vs 3.2M). The CNN's ~4.5M is the target of the backbone-freezing rule | CNN, CNN-HHO |  | train.py build_cnn() | `backbone-freezing` |  |  |
| `densenet121-activation` | `paper_deviation` | DenseNet121's final activation. | The last layer uses sigmoid for 32-way classification | Softmax | Sigmoid doesn't make sense for a single-label, mutually-exclusive classification problem, and every other model in the paper uses softmax for the same task; the sigmoid mention is treated as a slip in the writing |  | DenseNet121, DenseNet121-HHO |  |  |  |  |  |
| `densenet121-head` | `paper_deviation` | DenseNet121's classification head. | Fig. 6: MaxPooling2D, BatchNormalization, Dropout (0.3), Flatten, Dense (512 units, ReLU), Dropout (0.5), Dense (32 units, Sigmoid) | MaxPooling2D, BatchNormalization, Flatten, Dense (512, ReLU), Dropout, Dense (32, softmax): no Dropout (0.3) before Flatten. The remaining dropout is 0.5 for DenseNet121 and HHO-tuned for DenseNet121-HHO |  |  | DenseNet121, DenseNet121-HHO |  | train.py resizing_backbone() densenet121 head | `densenet121-activation` |  |  |
| `learning-rate` | `guess` | Learning rate for the models whose rate the paper omits. | The paper names an optimizer (Adam or Adamax) for each but never gives a rate | The framework default for ResNet50, DenseNet121, and DenseNet201; 1e-4 for the plain CNN | For the CNN, Adam's default (1e-3) was tried first and collapsed to predicting the class prior; 1e-4 is the value CNN-HHO's own search converged to |  | CNN, ResNet50, DenseNet121, DenseNet201 |  | train.py make_model_specs() comment on the CNN spec |  |  |  |
| `resnet50-dropout` | `guess` | The ResNet50 head's dropout rate. | Figure 5 just says "Flatten → Dropout → Dense" with no number | 0.5 | A plain default, distinct from the two dropout rates the paper does give for other models (0.6 for EfficientNet, 0.8 for DenseNet201) |  | ResNet50 |  |  |  |  |  |
| `metric-averaging` | `guess` | How precision, recall, and F1 are averaged across the 32 classes. | Not stated | Macro averaging (unweighted mean across classes) | The standard choice when it isn't specified, and consistent with the paper's own numbers: its F1 is sometimes lower than both its precision and recall, which can't happen with micro-averaging but is normal under macro or per-class averaging on an imbalanced label set |  |  |  |  |  |  |  |
| `split` | `guess` | The train/val/test split. | 64/16/20, stratified; the split itself and a seed are not published | Stratified 64/16/20, seed 42 |  |  |  |  |  |  |  | `arasl2018-grayscale` |
| `checkpoint-selection` | `guess` | Which evaluation Table II reports. | "Table II presents the best test results for each model" (§IV); it does not say best over what (epochs, runs, or HHO candidates) | One test evaluation at the end of the epoch count Table II lists; no checkpoint or run selection |  |  | All eight models |  |  |  |  |  |
| `results-prose-values` | `paper_internal_conflict` | Accuracies quoted in the results prose. | §IV's prose gives EfficientNet-B3 98.55%, ResNet50 97.90%, and CNN 94.50% with F1 94.2%; none of these appear in Table II | Table II's values as the targets |  |  |  |  |  |  |  |  |
| `hho-search` | `guess` | The HHO search budget. | HHO tunes "the learning rate, batch size, number of filters, and dropout rates" for CNN-HHO (§III.B.6, Fig. 7) and "learning rate, batch size, and dropout, to maximize validation accuracy" for DenseNet121-HHO (§III.B.7); no search ranges, population size, number of iterations, or cost per candidate evaluation | Search ranges: log10 learning rate [-4, -2], batch size 16-64, dropout 0.2-0.6, and for CNN-HHO a filter-count scale 0.5-1.5. Population 6, 4 iterations; each candidate judged by training it for 3 epochs on a 25% slice of the training data | Evaluating every candidate with a full training run would cost far more than the 5-epoch final run the paper reports; a fairly standard cheap-proxy setup | DenseNet121-HHO's search settled on a configuration that does slightly worse than the plain DenseNet121 default (95.11% vs. 96.39%), reported as-is rather than re-running the search until it beats the baseline | CNN-HHO, DenseNet121-HHO |  | train.py make_model_specs() hho_bounds; --hho-population/--hho-iterations/--hho-proxy-epochs/--hho-proxy-fraction defaults |  |  |  |

## Noncompliance

| ID | Description | Requirement | Run IDs | Effect | Resolution |
| --- | --- | --- | --- | --- | --- |
| `first-attempt-overwritten` | The first set of eight full runs (whole backbone frozen; 2026-08-30 06:03-07:18 UTC; recorded in commit aaa7d1a) was dropped from `runs` when the second set (partial unfreezing) was recorded under the same run IDs as attempt 1, and the second set overwrote its artifacts at the same paths on the results volume. | `reproduction.json.runs` records every meaningful retained attempt (AGENTS.md, evidence and completion contract) |  | The first attempt's test accuracies (CNN 96.28, CNN-HHO 87.32, EfficientNet-B0 91.09, EfficientNet-B3 86.85, ResNet50 81.09, DenseNet201 70.72, DenseNet121 85.74, DenseNet121-HHO 81.43) and artifact hashes survive only in commit aaa7d1a; its raw outputs are gone | Open: the run entries can be restored from commit aaa7d1a; the artifacts cannot |
| `failed-attempts-removed` | Commit a3cb787 deleted the README's "What went wrong along the way" section (CNN and CNN-HHO runs at chance before the learning-rate change, EfficientNet runs killed by hand, a ResNet50 run that finished at 33.7%, reruns that reused or resumed from broken outputs until volume folders were deleted) and its "Contacting the authors" section (no contact made). | README.md reports every failed attempt and any author contact (AGENTS.md, reporting integrity) |  | These dead ends and the no-contact statement are absent from the current report | Open: both sections can be restored from commit aaa7d1a |
