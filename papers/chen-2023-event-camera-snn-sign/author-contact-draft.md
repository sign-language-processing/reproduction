# Author-contact email — DRAFT, NOT SENT

Status: drafted 2026-10-05 after the independent attempt was complete, at the user's request. It has not been sent. Send it from a personal mail client outside this repository, then rename this file to `author-contact.md` and record the send date in `reproduction.json.author_contact`.

Point 4 is a request for author-held data. Under the study rules that is coordinated with Team S, so confirm with Team S before sending, or remove point 4 and let Team S send it separately.

**To:** li.su@cnu.edu.cn (corresponding author)
**Subject:** Reproducing Table 2 of "Sign Language Gesture Recognition and Classification Based on Event Camera with Spiking Neural Networks" (Electronics 2023, 12, 786)

---

Dear Prof. Su, dear Ms. Chen and co-authors,

We are carrying out an independent reproducibility study of sign-language-processing research. One of the papers in the study is your article *"Sign Language Gesture Recognition and Classification Based on Event Camera with Spiking Neural Networks"* (Electronics 2023, 12, 786). We are trying to reproduce Table 2.

Thank you for releasing the event data at github.com/najie1314/DVS. We could not find a code release, so we rebuilt the method from the paper. We used the published `master` split (600 training and 150 test files over 15 classes) and the Table 3 settings for the 77% row: step 80, dt 40 ms, Vth 0.3, lr 1e-3, batch 20, 200 epochs, the Eq. 8 decay and the Eq. 6 loss. The LIF neuron update follows your Algorithm 1.

Some details are not stated in the paper, so we ran four variants. Accuracy is on the 150 test files, with seed 1:

| Optimizer | 4x4 input pooling | Final epoch (200) | Best epoch |
|---|---|---|---|
| SGD, momentum 0 | average | 6.7% (no output spikes) | 6.7% |
| SGD, momentum 0 | max | 6.7% (no output spikes) | 6.7% |
| Adam | average | 74.7% | 77.3% (epoch 46) |
| Adam | max | 63.3% | 74.0% (epoch 51) |

With plain SGD the network never fires, so it does not train. Adam with average pooling comes close to your 77%, especially at its best epoch. We would like to check whether that matches what you did rather than guess. We would be very grateful for your help with the following points. The values we used are in brackets.

## 1. Code and training

- Is the training code available, or could you share it? [none found]
- Which optimizer and settings produced the 77% (and 68%) results? The paper says "SGD using standard settings". [plain SGD; Adam as a variant]
- Is the reported accuracy from the final epoch or the best test epoch? [both reported above]
- Which random seed or seeds did you use, and was each number a single run? [seed 1, one run]
- Was Table 2's 77.00% rounded to a whole percent? On 150 test files the nearest attainable values are 76.67% and 77.33%.

## 2. Network (Figure 3)

- What are the exact layers? We read Figure 3 as: 4x4 pooling (128 to 32), conv 2 to 34, LIF, conv 34 to 64, LIF, 2x2 pooling, conv 64 to 128, LIF, 2x2 pooling, FC 8192 to 256, LIF, FC 256 to 15, LIF. [3x3 kernels, padding 1]
- Is each pooling layer max or average? Is there a spiking (LIF) layer after the pooling layers?
- What are the LIF decay constant (tau) and the surrogate-gradient width? [0.25 and 0.5]
- Does a neuron fire when u > Vth, or when u − Vth > Vth? Some public STBP code effectively uses the second. [second]

## 3. Event encoding and evaluation

- How are events turned into the step x dt input frames? Are both polarities kept as two channels, and are events after step x dt dropped? [binary frame per 40 ms bin, two polarity channels, first 3.2 s only]
- The paper mentions a "validation set has 100". Which files form it, and was it used for training or model selection?
- How are the "first part" and "second part" of the test set defined for Acc1 and Acc2?

## 4. DAVIS346 recordings (DVS_Sign)

- The `main` branch named in the Data Availability Statement contains three classes whose files are identical to the v2e data on `master`. Would you be willing to share the DAVIS346 recordings used for the 68% result?
- If so, under what terms? Please also tell us whether the volunteers' consent covers research use and processing on cloud infrastructure.

We will report your answers, and what they change, in our study, with acknowledgement. Thank you very much for your time.

Kind regards,
Carlos Escolano
(on behalf of the REPRO-SIGN reproducibility study)
