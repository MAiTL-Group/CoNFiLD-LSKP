"""Portable entry points around the preserved training and evaluation code."""

import argparse
import importlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import yaml

from .config import ROOT, activate_imports, load_config, load_recipe, require_files, resolve_path


def run_legacy(relative, arguments):
    """Execute a bundled entry point from the release root, restoring caller state."""
    previous_argv, previous_cwd = sys.argv[:], Path.cwd()
    activate_imports()
    try:
        os.chdir(str(ROOT))
        sys.argv = [str(ROOT / relative)] + list(arguments)
        runpy.run_path(str(ROOT / relative), run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(str(previous_cwd))


def run_recipe(args):
    """Resolve one recipe, check inputs, save its effective values, then execute."""
    stage = {"train-cnf": "cnf", "fit-prior": "prior", "train-diffusion": "diffusion"}[args.command]
    cfg = load_recipe(args.config, stage, model=getattr(args, "model", None),
                      chart=getattr(args, "chart", None), training_seed=getattr(args, "train_seed", None),
                      overrides=getattr(args, "overrides", None))
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    if args.command == "train-cnf":
        needed = [cfg["data_path"]] + ([cfg["coor_path"]] if cfg.get("coor_path") else [])
        entry = "ConditionalNeuralField/scripts/train.py"
        Path(cfg["save_path"]).mkdir(parents=True, exist_ok=True)
    elif args.command == "fit-prior":
        needed = [cfg["train_npy"], cfg["valid_npy"]]
        entry = "Koopman/lskp/fit_prior.py"
    else:
        needed = [cfg["train_data_path"], cfg["valid_data_path"]]
        koopman = cfg.get("koopman", {})
        if koopman.get("enabled"):
            needed.append(koopman["checkpoint_path"])
        if cfg.get("resume_checkpoint"):
            needed.append(cfg["resume_checkpoint"])
        entry = "UnconditionalDiffusionTraining_and_Generation/scripts/train.py"
    require_files(needed)
    output = ROOT / "outputs/resolved_recipes"
    output.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=args.command + "_" + Path(args.config).stem + "_",
                                          suffix=".yaml", dir=str(output))
    recipe = Path(filename)
    with os.fdopen(descriptor, "w") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    arguments = ["--config", str(recipe)] if args.command == "fit-prior" else [str(recipe)]
    run_legacy(entry, arguments)


def extract_latents(args):
    """Use the paper's fixed temporal partitions with an explicit CNF checkpoint."""
    cfg = load_config("configs/data/{}.yaml".format(args.case))
    checkpoint = resolve_path(args.checkpoint)
    require_files([checkpoint, cfg["raw_data_npy"]])
    out = ROOT / "data/latents" / args.case / args.chart
    if (out / "split_meta.json").exists():
        raise FileExistsError("Latent dataset already exists: {}".format(out))
    values = ["--ckpt", str(checkpoint), "--out-root", str(out)]
    for key, value in cfg.items():
        values.extend(["--" + key.replace("_", "-"), str(value)])
    run_legacy("basicutility/trans_to_latents.py", values)


def evaluate(args, remaining):
    """Compute the unchanged case-specific metrics without producing figures."""
    from .config import activate_imports
    activate_imports()
    module = importlib.import_module("evaluation." + args.case)
    dns = resolve_path(args.dns_fields or "data/latents/{}/sn/test_raw_segment.npy".format(args.case))
    coords = resolve_path(args.coords or "data/raw/{}/coords.npy".format(args.case))
    require_files([dns] + ([coords] if args.case != "case2" else []))
    extra = ["--dns-fields", str(dns)]
    if args.case != "case2":
        extra += ["--coords", str(coords)]
    previous, cwd = sys.argv[:], Path.cwd()
    try:
        os.chdir(str(ROOT))
        sys.argv = [module.__file__] + extra + remaining
        module.main()
        parsed = module.parse_args()
        output = Path(parsed.out_dir)
        atoms = json.loads((output / "core_atoms.json").read_text())
        import numpy as np
        from evaluation.time_diagnostics import bootstrap_core_atoms
        bootstrap = bootstrap_core_atoms({key: np.asarray(value) for key, value in atoms.items()},
                                         n_bootstrap=args.bootstrap_samples, seed=parsed.seed)
        (output / "bootstrap_core.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    finally:
        sys.argv = previous
        os.chdir(str(cwd))


def build_pool(args):
    """Write a generation manifest for one horizon from completed seed directories."""
    import numpy as np
    runs, reference = [], None
    for directory in args.directories:
        path = resolve_path(directory)
        fields, latents, meta = (path / name for name in ("gen_fields.npy", "gen_latents.npy", "meta.json"))
        require_files([fields, latents, meta])
        codes = np.load(latents, mmap_mode="r", allow_pickle=False)
        field_array = np.load(fields, mmap_mode="r", allow_pickle=False)
        if codes.ndim != 3 or field_array.shape[0] != codes.shape[0] * codes.shape[1]:
            raise ValueError("Inconsistent field/latent sample dimensions: {}".format(path))
        current = (tuple(codes.shape[1:]), tuple(field_array.shape[1:]))
        if reference is not None and reference != current:
            raise ValueError("Pool entries must share a horizon and spatial layout")
        reference = current
        runs.append({"gen_fields": str(fields), "gen_latents": str(latents), "meta": str(meta)})
    out = resolve_path(args.out)
    if out.exists():
        raise FileExistsError(str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(out)


def parser():
    """Build the CLI without importing PyTorch or initializing CUDA."""
    result = argparse.ArgumentParser(description="CoNFiLD-LSKP training, sampling, and numerical evaluation")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("train-cnf", "fit-prior", "train-diffusion"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--dry-run", action="store_true", help="Print effective settings without executing")
        child.add_argument("--set", dest="overrides", action="append", default=[],
                           help="Override dotted.key=YAML-value after selecting a variant; repeat as needed")
        if name == "train-cnf":
            child.add_argument("--chart", choices=("origin", "sn"), help="Default: SN base recipe")
        elif name == "train-diffusion":
            child.add_argument("--model", choices=("origin", "sn", "lskp"), help="Default: LSKP base recipe")
            child.add_argument("--train-seed", type=int, help="Separate training seed and output directory")
    child = commands.add_parser("extract-latents")
    child.add_argument("--case", choices=("case1", "case2", "case3"), required=True)
    child.add_argument("--chart", choices=("origin", "sn"), required=True)
    child.add_argument("--checkpoint", required=True)
    child = commands.add_parser("sample")
    child.add_argument("--config", required=True)
    child.add_argument("--device", default="cuda")
    child.add_argument("--model", choices=("origin", "sn", "lskp"), help="Default: model in base recipe")
    child.add_argument("--dry-run", action="store_true", help="Print settings without loading weights or CUDA")
    for name in ("seed", "count", "horizon"):
        child.add_argument("--" + name, type=int)
    for name in ("checkpoint", "cnf-checkpoint", "cnf-normalizer", "latent-min", "latent-max", "output-dir"):
        child.add_argument("--" + name)
    child = commands.add_parser("evaluate", help="Other options are forwarded to the case-specific evaluator")
    child.add_argument("--case", choices=("case1", "case2", "case3"), required=True)
    child.add_argument("--dns-fields")
    child.add_argument("--coords")
    child.add_argument("--bootstrap-samples", type=int, default=1000)
    commands.add_parser("evaluate-events", add_help=False,
                        help="DNS-score event groups; options are forwarded to the event evaluator")
    commands.add_parser("analyze", add_help=False,
                        help="Supplementary numerical analyses; use analyze --help")
    child = commands.add_parser("build-pool")
    child.add_argument("--directories", nargs="+", required=True)
    child.add_argument("--out", required=True)
    return result


def main():
    """Dispatch commands; missing assets fail explicitly before long computation."""
    cli = parser()
    args, remaining = cli.parse_known_args()
    if remaining and args.command not in ("evaluate", "evaluate-events", "analyze"):
        cli.error("unrecognized arguments: " + " ".join(remaining))
    if args.command in ("train-cnf", "fit-prior", "train-diffusion"):
        run_recipe(args)
    elif args.command == "extract-latents":
        extract_latents(args)
    elif args.command == "sample":
        cfg = load_recipe(args.config, "inference", model=args.model)
        for arg, key in (("seed", "seed"), ("count", "test_batch_size"), ("horizon", "time_length")):
            if getattr(args, arg) is not None:
                cfg[key] = getattr(args, arg)
        for arg, key in (("checkpoint", "ema_path"), ("cnf_checkpoint", "cnf_checkpoint"),
                         ("cnf_normalizer", "cnf_normalizer"), ("latent_min", "min_val"),
                         ("latent_max", "max_val"), ("output_dir", "save_dir")):
            if getattr(args, arg) is not None:
                cfg[key] = str(resolve_path(getattr(args, arg)))
        if args.dry_run:
            print(yaml.safe_dump(cfg, sort_keys=False))
            return 0
        cnf_checkpoint = str(cfg.get("cnf_checkpoint") or "")
        if not cnf_checkpoint or "<" in cnf_checkpoint or ">" in cnf_checkpoint:
            cli.error("Set cnf_checkpoint or pass --cnf-checkpoint PATH to the CNF used for latent extraction")
        from .sampling import sample
        print(sample(cfg, args.device))
    elif args.command == "evaluate":
        if args.bootstrap_samples < 1:
            cli.error("--bootstrap-samples must be positive")
        evaluate(args, remaining)
    elif args.command == "evaluate-events":
        run_legacy("evaluation/event_cli.py", remaining)
    elif args.command == "build-pool":
        build_pool(args)
    elif args.command == "analyze":
        activate_imports()
        from evaluation.supplementary.cli import main as analyze
        return analyze(remaining)
    return 0
