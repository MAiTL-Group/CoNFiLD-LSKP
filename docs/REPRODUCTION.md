# Reproduction protocol

## Fixed data partitions

Indices are zero based and half open, referring to the retained sequence.
Windows use stride one and are formed separately within each partition.

| Case | Retained frames | Train | Validation | Test | Boundary gap | Window |
|---|---:|---|---|---|---:|---:|
| Case1 | 16000 | [0,12800) | [12927,14400) | [14527,16000) | 127 | 128 |
| Case2 | 1200 | [0,640) | [640,920) | [920,1200) | 0 | 256 |
| Case3 | 2880 | [0,2304) | [2304,2592) | [2592,2880) | 0 | 256 |

Train/validation/test window counts are 12673/1346/1346,
385/25/25 and 2049/33/33. Case1 has raw-frame offset one; Case2/3 use zero.
No extra time decimation is applied.

CNF snapshot reconstruction precedes this split. Prior standardization/PCA
and diffusion min/max use training latents only. Keep the CNF used for
extraction and decoding [aligned](CHECKPOINTS.md).

## Training and selection

Budgets and model variants are listed in [CONFIGURATIONS.md](CONFIGURATIONS.md).
Use the checkpoint selected by your run's validation loss.

The selector smooths finite validation diffusion loss with decay 0.90. The
first ten finite calls update smoothing without saving a best checkpoint;
subsequent updates require an improvement greater than 1e-4. Parameter EMA
has decay 0.9999. Validation cadence is 1000 steps for Case1 and LSKP,
and 100 steps for Case2/3 Origin and SN.

## Final priors

| Case | PCA dimension | Regimes | History descriptors | Ridge | Burst clustering boost |
|---|---:|---:|---|---:|---:|
| Case1 | 48 | 8 | yes | 1e-3 | 2.0 |
| Case2 | 48 | 1 | no | 1e-1 | 1.0 |
| Case3 | 16 | 1 | no | 1e-1 | 1.0 |

The frozen prior stores alignment, PCA, descriptor scales, operators, residual
variances and the burst threshold. Confidence-triggered weighting and Huber
penalties are configured in each diffusion recipe. At R=1, selective weights
are one while the stochastic transition penalty remains active.

## Evaluation and generation pools

Defaults use rollout seed 42 with 1/3/1 trajectories for Case1/2/3.
Optional pools use seeds 42/43/44 with the model fixed. For a horizon comparison,
generate each horizon independently in its own output directory.

| Case | Horizons | Final evaluation window / stride | Seed-42 windows |
|---|---|---|---:|
| Case1 | 1920, 2560, 3200, 3840 | 1473 / 368 | 8 |
| Case2 | 256, 384, 512, 768 | 280 / 70 | 24 |
| Case3 | 256, 384, 512, 768, 1024, 1536 | 288 / 72 | 19 |

The evaluator pools matched-length sliding windows, adding an end-aligned
window when needed. Fixed-lag distances stay within trajectory boundaries;
diagnostic features are standardized within each trajectory. Integral memory
uses normalized energy ACFs up to the first non-positive lag or 128 lags.

Read a run's aggregate metrics from `scorecard.json`. Window means differ because
Wasserstein distance and ACF truncation are nonlinear. Bootstrap intervals
resample window atoms against a fixed DNS reference; they measure window
variability, not independent-training variability. The training-seed study
instead reports mean and sample SD across three trained models per method.

See the [README](../README.md) for execution and
[SUPPLEMENTARY.md](SUPPLEMENTARY.md) for optional analyses.
