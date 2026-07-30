# Reproduction report: <paper short name>

**Citation:** <full citation>
**Paper:** <url> · **Code:** <url or "none published">
**Upstream commit:** `<sha>` · **Preference level:** <1 as-is | 2 patches | 3 reimplementation>
**Status:** `<reproducibility_status>`
**Reproduced by:** <name>, <date>

## Results

| Table | System | Dataset | Metric | Original | Reproduced | Δ |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

Copied baselines in the original table: <which, and whether we reproduced them ourselves>

## How to repeat this

```bash
docker build -t <tag> -f repositories/<USER>/<REPO>/Dockerfile .
docker run --rm --gpus all -v "$PWD/data:/data" <tag> bash scripts/train.sh
docker run --rm --gpus all -v "$PWD/data:/data" <tag> bash scripts/eval.sh
```

**Hardware:** <gpus> · **Runtime:** <h> · **GPU hours:** <h> · **Platform:** <local/s3it/modal>

## Data

Dataset, how it was obtained, license/permission basis.

## Patches

| Patch | Why it was necessary |
| --- | --- |
| `patches/01-....patch` | correctness reason |

None of these change model behavior. <Or: state exactly which does, and why it was unavoidable.>

## Guesses

Everything the paper did not specify, and what we chose instead.

| Detail | Paper says | We used | Rationale |
| --- | --- | --- | --- |

## Difficulties

What broke, what we tried, what dead ends cost time. Useful for the meta-analysis.

## Author contact

<None — reproduced from the paper and published code alone. / What we asked, when, and what it changed.>
