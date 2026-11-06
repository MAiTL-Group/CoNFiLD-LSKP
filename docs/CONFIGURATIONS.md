# Configurations and run variants

Each case has one YAML under each of `configs/cnf/`, `configs/data/`,
`configs/prior/`, `configs/diffusion/` and `configs/inference/`.
Paths are relative to the code directory. Use `--dry-run` to inspect resolved
settings and repeated `--set dotted.key=value` options for overrides.

## CNF charts

`--chart sn` uses the base SN model; `--chart origin` disables `NF.use_sn`.
Outputs go to `outputs/cnf/<case>/<chart>/`.

| Case | SN epochs | Origin epochs |
|---|---:|---:|
| Case1 | 9805 | 9805 |
| Case2 | 14301 | 5701 |
| Case3 | 10000 | 10000 |

```bash
python scripts/run.py train-cnf --config configs/cnf/case2.yaml \
  --chart origin --dry-run
```

SN and LSKP share the SN chart; Origin uses the Origin chart. Finish any CNF
continuation before choosing the checkpoint for extraction and decoding.

## Diffusion models

Use `--model origin`, `--model sn` or `--model lskp` with the per-case YAML:

| Case | Model | Latent chart | Maximum training steps | Validation every | Prior |
|---|---|---|---:|---:|---|
| Case1 | origin | origin | 1000000 | 1000 | off |
| Case1 | sn | sn | 360000 | 1000 | off |
| Case1 | lskp | sn | 255000 | 1000 | on |
| Case2 | origin | origin | 355000 | 100 | off |
| Case2 | sn | sn | 185000 | 100 | off |
| Case2 | lskp | sn | 315000 | 1000 | on |
| Case3 | origin | origin | 500000 | 100 | off |
| Case3 | sn | sn | 155000 | 100 | off |
| Case3 | lskp | sn | 270000 | 1000 | on |

The model flag selects the chart, budget, cadence and prior settings while
keeping the shared architecture, optimizer and selector. It also sets:

- Inputs: `data/latents/<case>/<chart>/{train,valid}_data.npy`.
- Run directory: `outputs/diffusion/<case>/<model>/`.
- Min/max: `minmax/latent_min.npy` and `minmax/latent_max.npy` inside that run.

```bash
python scripts/run.py train-diffusion --config configs/diffusion/case3.yaml \
  --model sn --dry-run
```

## Additional Case3 training seeds

The default training seed is 42. `--train-seed N` sets another seed and adds
`_seedN` to the run directory and min/max paths. All other settings stay fixed.
Even an explicit `--train-seed 42` adds the suffix; omitting the flag keeps
the base run path. The supplementary study uses seeds 42, 1042 and 2042.

```bash
python scripts/run.py train-diffusion --config configs/diffusion/case3.yaml \
  --model sn --train-seed 1042 --dry-run
python scripts/run.py train-diffusion --config configs/diffusion/case3.yaml \
  --model lskp --train-seed 2042 --dry-run
```

## Inference models and rollout seeds

`--model` selects the CNF architecture; `--seed` selects rollout randomness.
Defaults use `outputs/diffusion/<case>/<model>/` and
`outputs/cnf/<case>/<chart>/`. Choose your CNF checkpoint explicitly.
For a different trained run, override its EMA and saved min/max:

```bash
RUN=outputs/diffusion/case3/sn_seed1042
CNF="outputs/cnf/case3/sn/checkpoint_<epoch>.pt"
python scripts/run.py sample --config configs/inference/case3.yaml --model sn \
  --checkpoint "$RUN/ema_0.9999_best.pt" \
  --cnf-checkpoint "$CNF" \
  --cnf-normalizer outputs/cnf/case3/sn/normalizer_params.pt \
  --latent-min "$RUN/minmax/latent_min.npy" \
  --latent-max "$RUN/minmax/latent_max.npy" \
  --output-dir outputs/samples/case3/sn_seed1042 --seed 42 --dry-run
```

Replace `<epoch>` with the CNF used for this run's extraction. Remove
`--dry-run` to generate after checking the [file alignment](CHECKPOINTS.md).
Origin requires `--model origin` and its own CNF; SN/LSKP use the SN chart.

## Data and priors

Data YAMLs fix temporal splits, offsets, window lengths and boundary gaps.
Prior YAMLs select the SN latent inputs and case-specific PCA/regime settings.
Numerical values are in [REPRODUCTION.md](REPRODUCTION.md).
Resolved training recipes are saved under `outputs/resolved_recipes/`.

## Resume training

Use a full checkpoint with `--set resume_checkpoint=PATH`,
`--set resume_optimizer=true` and `--set resume_ema=true`. Its step is restored;
`resume_step_override` explicitly overrides it. EMA-only sampling weights
contain no optimizer/RNG state.
