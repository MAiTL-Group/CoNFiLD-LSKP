# Supplementary experiments and numerical analyses

These optional analyses use your trained models and generated samples, writing
`summary.json` and, where applicable, `rows.csv`. All paths, including those
inside manifests, are relative to the repository root. Use a new output
directory per analysis and load only trusted checkpoints.

## 1. Recipe variants

Training and prior fitting accept repeated `--set dotted.key=value` options,
applied after `--model`/`--train-seed`. Use `--dry-run` to inspect them.
When changing a diffusion run's destination, set all three output paths:

```bash
run=outputs/diffusion/case1/no_history
python scripts/run.py train-diffusion --config configs/diffusion/case1.yaml \
  --set log_path="$run" \
  --set min_val="$run/minmax/latent_min.npy" \
  --set max_val="$run/minmax/latent_max.npy" \
  --set koopman.checkpoint_path=outputs/prior/case1_no_history/lskp_prior.pt \
  --dry-run
```

Remove `--dry-run` and select a GPU to train.

### Case1 component and regime-count variants

Apply these changes to `configs/prior/case1.yaml`, with a distinct `save_dir`
for each fit:

| Variant | Prior-fit changes |
|---|---|
| No history | `include_prev_delta=false`, `include_delta2=false`, `artifact_version=lskp13_no_history_ablation` |
| Single-regime component ablation | `num_regimes=1`, `burst_kmeans_boost=1.0`, `artifact_version=lskp13_single_regime_ablation` |
| Regime sweep | `num_regimes=R`, `artifact_version=lskp13_regime_sensitivity_rR`, for R=1,2,4,16 |
| R=8 reference | Reuse your base fitted prior and LSKP run |

The R=1 sweep keeps `burst_kmeans_boost=2.0`, unlike the component ablation's
1.0. `burst_regime_count` defaults to zero.

```bash
python scripts/run.py fit-prior --config configs/prior/case1.yaml \
  --set include_prev_delta=false --set include_delta2=false \
  --set artifact_version=lskp13_no_history_ablation \
  --set save_dir=outputs/prior/case1_no_history

for r in 1 2 4 16; do
  python scripts/run.py fit-prior --config configs/prior/case1.yaml \
    --set num_regimes="$r" --set artifact_version="lskp13_regime_sensitivity_r$r" \
    --set save_dir="outputs/prior/case1_r$r"
done
```

Train each diffusion with the Case1 LSKP base (255k steps), its fitted prior
and separate log/min/max paths. For no-reliability runs, retain the base prior
and set:

```text
--set koopman.selective_conf_threshold=-1.0
--set koopman.selective_weight_scale=1.0
--set koopman.selective_burst_conf_threshold=-1.0
--set koopman.selective_burst_scale=1.0
```

Use the same overrides with Case2/3's own recipes. At R=1 these factors are
already inactive.

### Generic temporal smoothness controls

Use the Case1 base with the complete mapping below. Keep its 255k budget and
other training/evaluation settings fixed.

```bash
run=outputs/diffusion/case1/velocity_l2
python scripts/run.py train-diffusion --config configs/diffusion/case1.yaml \
  --set log_path="$run" \
  --set min_val="$run/minmax/latent_min.npy" \
  --set max_val="$run/minmax/latent_max.npy" \
  --set 'koopman={enabled: false}' \
  --set 'generic_regularizer={enabled: true, mode: velocity_l2, warmup_steps: 5000, interval: 4, subbatch: 4, window_size: 128, sigma_max: 0.6, lambda_base: 0.02, ratio_cap_vs_diff: 0.03, ramp_steps: 0, normalize_by_signal: false, scale_eps: 1.0e-8}' \
  --dry-run
```

For acceleration, use `mode: acceleration_l2` and another run directory.
Neither control needs a fitted prior.

### Chart/prior factorial

The main variants cover Origin without prior, SN without prior and SN with
prior. For **Origin with prior**, fit a prior on that chart's train/valid
latents, then use the **LSKP diffusion base**, overriding its train/valid paths,
prior and log/min/max destinations. `train-diffusion --model origin` disables
the prior and changes the budget/cadence, so use the LSKP settings for this cell.

```bash
python scripts/run.py fit-prior --config configs/prior/case3.yaml \
  --set train_npy=data/latents/case3/origin/train_data.npy \
  --set valid_npy=data/latents/case3/origin/valid_data.npy \
  --set artifact_version=origin_case3_lskp13r1 \
  --set save_dir=outputs/prior/case3_origin
```

For sampling, `--model origin` selects the Origin CNF architecture. Supply all
five model inputs and a separate output directory, using the exact CNF that
provided this cell's training latents; see [CHECKPOINTS.md](CHECKPOINTS.md).

## 2. Generate and evaluate each trained cell

Bind each run's own EMA, min/max and matching CNF. Replace `<epoch>` with the
CNF used for extraction:

```bash
run=outputs/diffusion/case3/lskp_seed1042
CNF="outputs/cnf/case3/sn/checkpoint_<epoch>.pt"
for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run.py sample \
    --config configs/inference/case3.yaml --model lskp \
    --checkpoint "$run/ema_0.9999_best.pt" \
    --latent-min "$run/minmax/latent_min.npy" \
    --latent-max "$run/minmax/latent_max.npy" \
    --cnf-checkpoint "$CNF" \
    --cnf-normalizer outputs/cnf/case3/sn/normalizer_params.pt \
    --seed "$seed" --count 1 --horizon 1536 \
    --output-dir outputs/supplementary/lskp_seed1042
done

python scripts/run.py build-pool \
  --directories outputs/supplementary/lskp_seed1042/seed_0042 \
    outputs/supplementary/lskp_seed1042/seed_0043 outputs/supplementary/lskp_seed1042/seed_0044 \
  --out outputs/supplementary/lskp_seed1042_pool.json

python scripts/run.py evaluate --case case3 \
  --gen-manifest outputs/supplementary/lskp_seed1042_pool.json \
  --paired-policy window_pool --bootstrap-samples 200 \
  --out-dir outputs/supplementary/lskp_seed1042_metrics
```

`sample` adds its own `seed_XXXX` directory. The Case3 pool has 57 windows from
three 1536-frame rollouts. Use 200 bootstrap resamples for training-seed
summaries, 1000 for the common-step grid, and full defaults for other analyses.

### Training-seed summary

Train SN and LSKP at seeds 42,1042,2042 with `--train-seed`, keeping the CNF
and fitted prior fixed. Use their base budgets, cadences and selectors.
Sample/evaluate all six cells, then list their `bootstrap_core.json` files:

```json
{"runs": [
  {"model":"sn", "train_seed":42, "metrics":"outputs/supplementary/sn_seed42_metrics/bootstrap_core.json"},
  {"model":"sn", "train_seed":1042, "metrics":"outputs/supplementary/sn_seed1042_metrics/bootstrap_core.json"},
  {"model":"sn", "train_seed":2042, "metrics":"outputs/supplementary/sn_seed2042_metrics/bootstrap_core.json"},
  {"model":"lskp", "train_seed":42, "metrics":"outputs/supplementary/lskp_seed42_metrics/bootstrap_core.json"},
  {"model":"lskp", "train_seed":1042, "metrics":"outputs/supplementary/lskp_seed1042_metrics/bootstrap_core.json"},
  {"model":"lskp", "train_seed":2042, "metrics":"outputs/supplementary/lskp_seed2042_metrics/bootstrap_core.json"}
]}
```

```bash
python scripts/run.py analyze training-seeds \
  --manifest outputs/supplementary/training_seeds.json \
  --out-dir outputs/analysis/training_seeds
```

Each file must have 57 window atoms. The summary reports mean and sample SD
(`ddof=1`) across trained models, paired LSKP-minus-SN differences and win counts,
with each cell's window interval separate.

### Common-step grid

Evaluate all three Case3 models at 50k/100k/155k using exact saved step EMAs,
rollout seeds 42/43/44, T=1536 and 1000 bootstrap resamples. For example, pass
`--checkpoint path/to/ema_0.9999_050000.pt` and a distinct output directory.
Missing steps require the actual saved checkpoints, not nearby replacements.

Include all nine model-step cells in a manifest with the schema
`{"runs":[{"model":"origin","step":50000,"metrics":"outputs/.../bootstrap_core.json"}, ...]}`.

```bash
python scripts/run.py analyze common-steps \
  --manifest outputs/supplementary/common_steps.json \
  --out-dir outputs/analysis/common_steps
```

This compares fixed steps while retaining each run's validation cadence.

### Other comparisons

`analyze metric-grid` takes records with `model`, optional unique `label`, and
`metrics`. Use it for factorial, component, smoothness, R-sweep or horizon
summaries. Supply every planned cell and label its case, horizon and budget.

For Case2 horizon stress, generate T=768/1024/1280 independently with the same
checkpoints and three rollout seeds, then use the Case2 window evaluator.
`evaluate-events` uses DNS-score event groups, distinct from LSKP regimes.

## 3. Offline numerical diagnostics

These tasks run on CPU. Default latent paths are
`data/latents/CASE/sn/{train,valid,test}_data.npy` and the prior directory is
`outputs/prior/CASE`. Override with `--prior-dir`, `--train`, `--valid`, `--test`.

```bash
python scripts/run.py analyze pca --case case3 --out-dir outputs/analysis/case3_pca
python scripts/run.py analyze input-gradients --case case3 --out-dir outputs/analysis/case3_gradients
python scripts/run.py analyze descriptor-coherence --case case1 --out-dir outputs/analysis/case1_coherence
python scripts/run.py analyze prior-health --case case2 --out-dir outputs/analysis/case2_prior_health
```

| Task | Inputs and outputs |
|---|---|
| `pca` | Train-fitted dimensions 8,16,32,48,64,96; covariance retention, train-centered validation energy, one-step MSE and operator conditioning. |
| `input-gradients` | Up to 4096 evenly spaced validation triplets; latent-input gradients versus velocity/acceleration L2. Requires R=1. |
| `descriptor-coherence` | Frozen soft prior, hard Lloyd, global R=1, five frequency-matched random-label and five circular-shift controls (seeds 3407-3411). Training-only operator fits; descriptive SD uses ddof=0. |
| `prior-health` | Support by split, operator Frobenius norm, noise-floor fraction and validation whitened residual. Requires the matching `prior_reference_metrics.json` from `fit-prior`. |

For prior-health sensitivity, vary `num_regimes` and `lambda_k_ridge` with
`--set`, saving each full-split fit separately.

The following tasks also need a root containing `seed_*/gen_latents.npy`:

```bash
python scripts/run.py analyze tube-calibration --case case1 \
  --generated outputs/samples/case1/lskp --out-dir outputs/analysis/case1_tubes
python scripts/run.py analyze reliability-gate --case case1 \
  --generated outputs/samples/case1/lskp --out-dir outputs/analysis/case1_gate
python scripts/run.py analyze amplification --case case1 \
  --generated outputs/samples/case1/lskp --out-dir outputs/analysis/case1_amplification
python scripts/run.py analyze memorization --case case1 \
  --generated outputs/samples/case1/lskp --out-dir outputs/analysis/case1_nearest_neighbor
```

Latents have shape `[B,T,D]`, normally B=1 for Case1/3 and B=3 for Case2.
Use matching horizons/shapes. Multi-regime amplification requires B=1 per
seed; the other diagnostics and R=1 amplification retain all samples.

- Tube calibration reports residual RMS, coverage, confidence association and support using the frozen prior.
- Reliability uses thresholds 0.15/0.25 and the train-fitted burst threshold. DNS intervals use whole windows and contiguous 32-window blocks, 1000 resamples, seeds 3407/3408. Generated groups are descriptive.
- Amplification uses product lengths 1,4,8,16; R=1 uses exact powers, while multi-regime products use seed42 and up to 2048 starts per seed file.
- Nearest-neighbor checks use disjoint generated windows, up to 256 test windows, exact RMS distances and threshold 1e-4 in latent and coarse coordinates.

### Logged selector trajectories and coupling scale

```bash
python scripts/run.py analyze validation-trajectory \
  --log outputs/diffusion/case3/lskp_seed1042/log.txt \
  --best-report outputs/diffusion/case3/lskp_seed1042/best_report.json \
  --out-dir outputs/analysis/lskp_seed1042_validation
python scripts/run.py analyze coupling-health \
  --config outputs/resolved_recipes/ACTUAL_SAVED_RECIPE.yaml \
  --run-dir outputs/diffusion/case3/lskp_seed1042 \
  --out-dir outputs/analysis/lskp_seed1042_coupling
```

Use the saved effective recipe with its actual overrides. Validation summaries
report observed losses, best updates and resume events. Coupling health uses
`lambda_eff * raw_prior_loss / training_mse` as a logged scale proxy.

### Tube-score / temporal-metric association

List both SN and LSKP for each case-horizon pair. Use seed42 aggregate
`scorecard.json` metrics and same-horizon seed files for pooled tube RMS:

```json
{"runs":[
 {"case":"case1","horizon":1920,"model":"sn","prior":"outputs/prior/case1/lskp_prior.pt","generated":"outputs/samples/case1/sn_T1920","metrics":"outputs/metrics/case1/sn_T1920_seed42/scorecard.json"},
 {"case":"case1","horizon":1920,"model":"lskp","prior":"outputs/prior/case1/lskp_prior.pt","generated":"outputs/samples/case1/lskp_T1920","metrics":"outputs/metrics/case1/lskp_T1920_seed42/scorecard.json"}
]}
```

Expand to all 14 pairs: Case1 four horizons, Case2 four and Case3 six,
listed in [REPRODUCTION.md](REPRODUCTION.md). Both models in a pair use the
same frozen scoring prior. Outputs include absolute LSKP-minus-SN differences,
Pearson/Spearman association, case-only/leave-one-case-out checks and pair
bootstrap (1000 draws, seed3407).

```bash
python scripts/run.py analyze tube-association \
  --manifest outputs/supplementary/tube_association.json \
  --out-dir outputs/analysis/tube_association
```

Results record pair count and `paper_grid_complete` for the 14-pair grid.

## 4. Required assets

Supply your trained checkpoints and evaluated metrics. Save fixed-step EMAs
during training if you plan to use the common-step analysis.
