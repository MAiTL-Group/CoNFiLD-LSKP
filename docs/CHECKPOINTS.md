# Keep your model files aligned

**Use the same CNF checkpoint for latent extraction and field decoding.**
CNF jointly learns the decoder and snapshot codes; a later decoder is not an
interchangeable replacement for an earlier checkpoint's codes.

After training a Case3 SN CNF, choose an actual saved epoch:

```bash
CNF="outputs/cnf/case3/sn/checkpoint_<epoch>.pt"
python scripts/run.py extract-latents --case case3 --chart sn --checkpoint "$CNF"
sha256sum "$CNF"
```

Keep this file fixed through prior fitting, diffusion training and generation.
SN and LSKP share the SN chart; Origin uses its own chart.

## Inputs for generation

For Case3 LSKP, supply all five options shown in the [README](../README.md):

| Option | File from your training run |
|---|---|
| `--checkpoint` | `outputs/diffusion/case3/lskp/ema_0.9999_best.pt` |
| `--cnf-checkpoint` | The exact `$CNF` used for extraction |
| `--cnf-normalizer` | `outputs/cnf/case3/sn/normalizer_params.pt` |
| `--latent-min` | `outputs/diffusion/case3/lskp/minmax/latent_min.npy` |
| `--latent-max` | `outputs/diffusion/case3/lskp/minmax/latent_max.npy` |

The inference defaults point to your training outputs. Set `cnf_checkpoint`
in the YAML or pass `--cnf-checkpoint` to replace `<CNF_CHECKPOINT>`.
`--model origin` selects the non-SN CNF architecture; `sn` and `lskp` select SN.
The prior is needed during LSKP training, not sampling.

Changing the CNF requires new latents, statistics, a refitted prior for LSKP and
retrained diffusion in separate output directories. Keep the path and checksum
with your run records. `sample --dry-run` prints resolved inputs; verify their
origin yourself before generation.

## Resume checkpoints versus sampling weights

Resume training with a full checkpoint containing optimizer/RNG state; sample
with the selected parameter EMA. See
[resume training](CONFIGURATIONS.md#resume-training).
