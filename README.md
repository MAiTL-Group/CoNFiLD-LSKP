# CoNFiLD-LSKP

[![Python 3.8.18](https://img.shields.io/badge/Python-3.8.18-blue.svg)](environment.yml)
[![PyTorch 2.2.0](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c.svg)](requirements.txt)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

Code for **DNS-Calibrated Local Stochastic Transition Closure for PDE-Free
Long-Horizon Turbulence Diffusion** (NeurIPS 2026).

## Overview

Plausible turbulence snapshots can accompany drift in temporal memory and
transition statistics over long rollouts. CoNFiLD-LSKP addresses this with a
local stochastic Koopman prior (LSKP) calibrated from direct numerical
simulation (DNS) data in a conditional neural field (CNF) latent space.
The prior is fitted offline on coarse observables of training latents,
combining transition operators with residual variances. During diffusion
training, the frozen prior penalizes whitened residuals of predicted-clean
latent transitions through a noise-gated loss. The CNF decoder and reverse
diffusion sampler remain unchanged.

The repository supports Origin-CoNFiLD, SN-CNF and LSKP across three flow
cases. This release provides code
for training, generation and evaluation; model checkpoints are not distributed.

## 1. Install

Use Linux, Conda and a CUDA-capable GPU. The environment pins Python 3.8.18;
metrics run on CPU.
Clone or download the code into `CoNFiLD-LSKP`, then install:

```bash
cd CoNFiLD-LSKP
conda env create -f environment.yml
conda activate confild-lskp
python -m pip check
python scripts/run.py --help
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The CUDA check must print `True` before training or generation.

## 2. Get data

Download **data.zip** from the [CoNFiLD data record](https://zenodo.org/records/14037782)
and arrange these arrays unchanged:

| Archive path | Destination |
|---|---|
| `data/Case1/data.npy` | `data/raw/case1/data.npy` |
| `data/Case1/coords.npy` | `data/raw/case1/coords.npy` |
| `data/Case2/output.npy` | `data/raw/case2/output.npy` |
| `data/Case3/data.npy` | `data/raw/case3/data.npy` |
| `data/Case3/coords.npy` | `data/raw/case3/coords.npy` |

Extraction applies the temporal split; keep all raw frames in their original order.

Run the Bash examples from `CoNFiLD-LSKP/`; paths are relative to this directory. Use `tmux` for long jobs over SSH.

## 3. Train a model

The example trains Case3 LSKP:

```bash
set -e
CASE=case3
MODEL=lskp
CHART=sn
if [ "$MODEL" = "origin" ]; then
  CHART=origin
fi
```

| Model | `MODEL` | `CHART` | Fit a prior |
|---|---|---|---|
| Origin-CoNFiLD | `origin` | `origin` | no |
| SN-CNF | `sn` | `sn` | no |
| LSKP | `lskp` | `sn` | yes |

Change `CASE` and `MODEL` for another run; the block selects the matching
chart automatically. SN and LSKP share one SN CNF and its latents per case.

First train a CNF:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py train-cnf \
  --config "configs/cnf/$CASE.yaml" --chart "$CHART"
```

Replace `<epoch>` with your chosen saved CNF epoch. **Use this exact checkpoint
for both latent extraction and final field decoding.**

```bash
CNF="outputs/cnf/$CASE/$CHART/checkpoint_<epoch>.pt"
python scripts/run.py extract-latents \
  --case "$CASE" --chart "$CHART" --checkpoint "$CNF"

if [ "$MODEL" = "lskp" ]; then
  CUDA_VISIBLE_DEVICES='' python scripts/run.py fit-prior \
    --config "configs/prior/$CASE.yaml"
fi

CUDA_VISIBLE_DEVICES=0 python scripts/run.py train-diffusion \
  --config "configs/diffusion/$CASE.yaml" --model "$MODEL"
```

Outputs are `data/latents/$CASE/$CHART/`,
`outputs/prior/$CASE/lskp_prior.pt` for LSKP, and
`outputs/diffusion/$CASE/$MODEL/` for diffusion checkpoints and min/max.
Keep the variables set for the next section. Reuse prepared CNF inputs when
training another diffusion model on the same chart. Add `--dry-run` to inspect
a training recipe; see [configurations](docs/CONFIGURATIONS.md) for overrides
and resumption.

## 4. Generate and evaluate

Use the same CNF checkpoint and your diffusion run's selected EMA and statistics:

```bash
RUN="outputs/diffusion/$CASE/$MODEL"
SAMPLES="outputs/samples/$CASE/$MODEL"
CUDA_VISIBLE_DEVICES=0 python scripts/run.py sample \
  --config "configs/inference/$CASE.yaml" --model "$MODEL" \
  --checkpoint "$RUN/ema_0.9999_best.pt" \
  --cnf-checkpoint "$CNF" \
  --cnf-normalizer "outputs/cnf/$CASE/$CHART/normalizer_params.pt" \
  --latent-min "$RUN/minmax/latent_min.npy" \
  --latent-max "$RUN/minmax/latent_max.npy" \
  --seed 42 --output-dir "$SAMPLES"
python scripts/run.py evaluate --case "$CASE" \
  --dns-fields "data/latents/$CASE/$CHART/test_raw_segment.npy" \
  --gen-fields "$SAMPLES/seed_0042/gen_fields.npy" \
  --gen-latents "$SAMPLES/seed_0042/gen_latents.npy" \
  --out-dir "outputs/evaluation/$CASE/${MODEL}_seed42" \
  --paired-policy window_pool --seed 42
```

Sampling writes `gen_latents.npy` (`sample,time,latent`), `gen_fields.npy`
(flattened sample/time axes) and `meta.json`; evaluation writes `scorecard.json`.
Use `--seed`, `--count`, `--horizon` and `--output-dir` to change rollout settings.
Sampling preserves existing outputs, so use a new directory for another run.
Training and sampling are stochastic; numerical results vary between runs.

Read this run's metrics:

```bash
cat "outputs/evaluation/$CASE/${MODEL}_seed42/scorecard.json"
```

The primary fields are `w1_time_lag_1`, `w1_time_lag_16`, `Tint_gap` and
`Err_logL2`; all are lower-is-better errors against the held-out DNS reference.

## Guides

| Topic | Guide |
|---|---|
| CNF and diffusion file alignment | [CHECKPOINTS.md](docs/CHECKPOINTS.md) |
| Model variants, seeds and resumption | [CONFIGURATIONS.md](docs/CONFIGURATIONS.md) |
| Data splits and metric protocol | [REPRODUCTION.md](docs/REPRODUCTION.md) |
| Optional supplementary analyses | [SUPPLEMENTARY.md](docs/SUPPLEMENTARY.md) |

## Project structure

Main directories and entry points:

```text
CoNFiLD-LSKP/
|-- ConditionalNeuralField/                    # CNF models and training
|-- basicutility/                             # Latent extraction and temporal splits
|-- Koopman/lskp/                              # Stochastic transition prior
|-- UnconditionalDiffusionTraining_and_Generation/
|   |-- scripts/train.py                      # Diffusion training entry point
|   `-- src/                                  # Denoiser, diffusion and regularization
|-- configs/                                  # Per-case training and inference recipes
|-- lskp_release/                             # CLI, configuration and sampling
|-- evaluation/                               # Field and temporal diagnostics
|-- scripts/run.py                            # Unified command-line entry point
`-- docs/                                     # Detailed usage and protocols
```

| Component | Main implementation |
|---|---|
| CNF training | [`ConditionalNeuralField/scripts/train.py`](ConditionalNeuralField/scripts/train.py) |
| Latent extraction and temporal splits | [`basicutility/trans_to_latents.py`](basicutility/trans_to_latents.py) |
| Prior fitting | [`Koopman/lskp/fit_prior.py`](Koopman/lskp/fit_prior.py) |
| Prior representation and transition scores | [`Koopman/lskp/data.py`](Koopman/lskp/data.py) |
| Training-time prior penalty | [`UnconditionalDiffusionTraining_and_Generation/src/koopman_regularizer_v6.py`](UnconditionalDiffusionTraining_and_Generation/src/koopman_regularizer_v6.py) |
| Sampling and field decoding | [`lskp_release/sampling.py`](lskp_release/sampling.py) |
| Case-specific evaluation | [`evaluation/`](evaluation/) |
| Temporal metrics | [`evaluation/time_diagnostics.py`](evaluation/time_diagnostics.py) |

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{lin2026dnscalibrated,
  title     = {{DNS}-Calibrated Local Stochastic Transition Closure for {PDE}-Free Long-Horizon Turbulence Diffusion},
  author    = {},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License and attribution

Built on [CoNFiLD](https://github.com/jx-wang-s-group/CoNFiLD), Pan Du et al.,
*Conditional neural field latent diffusion model for generating spatiotemporal
turbulence*, Nature Communications 15, 10416 (2024),
[DOI](https://doi.org/10.1038/s41467-024-54712-1).

Our code additions use [Apache-2.0](LICENSE). Inherited notices are in
[NOTICE](NOTICE) and [third-party licenses](third_party_licenses/README.md).
External data and assets retain their own terms; see
[provenance](docs/PROVENANCE.md).
