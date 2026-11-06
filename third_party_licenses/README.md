# Third-Party Notices

These licenses apply to the respective inherited code portions.
Our code additions use [Apache-2.0](../LICENSE).

## Components and original licenses

| Component | Inherited portions | License text |
|---|---|---|
| [CoNFiLD](https://github.com/jx-wang-s-group/CoNFiLD) | Original CNF modules, training framework, utilities and diffusion implementation extended in this repository | [Apache-2.0](../LICENSE.upstream) |
| [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | Diffusion model, sampling, likelihood, resampling, mixed-precision, neural-network and training utilities inherited through CoNFiLD | [MIT](OpenAI-guided-diffusion-MIT.txt) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | The `AttentionPool2d` implementation credited in `UnconditionalDiffusionTraining_and_Generation/src/unet.py` | [MIT](OpenAI-CLIP-MIT.txt) |
| [OpenAI baselines](https://github.com/openai/baselines) | The logger and MPI-helper lineage credited in `UnconditionalDiffusionTraining_and_Generation/src/logger.py` | [MIT](OpenAI-baselines-MIT.txt) |

The diffusion utility paths include
`UnconditionalDiffusionTraining_and_Generation/src/{dist_util,fp16_util,gaussian_diffusion,logger,losses,nn,resample,respace,script_util,train_util,unet}.py`.
The earlier [Ho et al. implementation](https://github.com/hojonathanho/diffusion/tree/1e0dceb3b3495bbe19116a5e1b3596cd0706c543)
remains credited in the original `gaussian_diffusion.py`, `losses.py` and
`unet.py` source documentation. Existing notices and source links are retained.
The package dependencies installed with pip/Conda retain their own licenses;
their source distributions are not bundled here.

## License sources

The files preserve the original copyright and warranty notices:

- CoNFiLD: [LICENSE at 449835e](https://github.com/jx-wang-s-group/CoNFiLD/blob/449835e76a2b35f24f066e4aa769ea23d3018ab6/LICENSE).
- guided-diffusion: [LICENSE at 22e0df8](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/LICENSE).
- CLIP: [LICENSE at d05afc4](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE).
- baselines: [LICENSE at ea25b9e](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/LICENSE).

## Modifications

The following files carry modification notices.
Paths in the last six rows are relative to
`UnconditionalDiffusionTraining_and_Generation/src/`.

| File | Changes relative to CoNFiLD |
|---|---|
| `ConditionalNeuralField/cnf/components.py` | Spectral-normalized latent-conditioning layers |
| `ConditionalNeuralField/cnf/initialization.py` | Documentation of the retained initialization routines |
| `ConditionalNeuralField/cnf/nf_networks.py` | Spectral-normalized conditioning options and imports |
| `ConditionalNeuralField/scripts/train.py` | CNF training, seeding, configuration and logging |
| `UnconditionalDiffusionTraining_and_Generation/scripts/train.py` | Local prior integration, seeding, configuration and logging |
| `dist_util.py` | Single-process and optional distributed execution support |
| `fp16_util.py` | Additional training diagnostics |
| `gaussian_diffusion.py` | Predicted-clean outputs and validation loss support |
| `logger.py` | Local and optional experiment logging |
| `nn.py` | Checkpointed backward for repeated first-order gradient queries |
| `train_util.py` | Prior regularization, validation selection and training support |

Portable tools and evaluation packaging are described in
[PROVENANCE.md](../docs/PROVENANCE.md).

## External assets

Data and model archives retain their own licenses. Their sources, creators and
CC BY 4.0 attribution requirements are listed in
[PROVENANCE.md](../docs/PROVENANCE.md). No upstream endorsement is implied.
