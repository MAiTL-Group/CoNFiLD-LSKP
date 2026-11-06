"""Explicit-input CPU analyses for the supplementary evidence."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from lskp_release.config import ROOT, load_config, require_files, resolve_path
from .common import load_artifact, load_coarse_split, prior_from_dict
from .summaries import METRICS, summarize_metrics


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_result(output, payload):
    """Write new JSON/CSV only; never replace a previous analysis directory."""
    output = resolve_path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = _clean(payload)
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    rows = payload.get("rows", [])
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "rows.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(output / "summary.json")


def _paths(args):
    return {split: resolve_path(getattr(args, split) or
            "data/latents/{}/sn/{}_data.npy".format(args.case, split))
            for split in ("train", "valid", "test")}


def _generated(root, horizon=None):
    """Require consistent [batch,time,latent] seed files, including Case2 batches."""
    paths = sorted(root.glob("seed_*/gen_latents.npy"))
    if not paths:
        raise FileNotFoundError("No seed_*/gen_latents.npy under " + str(root))
    shape = None
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 2:
            raise ValueError("Supplementary rollout diagnostics require [B,T,D]: " + str(path))
        if horizon is not None and array.shape[1] != horizon:
            raise ValueError("Declared horizon differs from " + str(path))
        if shape is not None and tuple(array.shape) != shape:
            raise ValueError("Mixed rollout shapes in " + str(root))
        shape = tuple(array.shape)
    return paths


def prior_health(prior_dir, splits):
    """Measure fitted support and operator/noise scale on explicit train/valid data."""
    from .coherence import nearest_labels
    from Koopman.lskp.data import build_transition_descriptor, normalize_descriptor
    artifact, cfg = load_artifact(prior_dir, {s: splits[s] for s in ("train", "valid")})
    prior = prior_from_dict(artifact["local_stochastic_prior"])
    regimes = int(prior.centers.shape[0])
    counts = {}
    for split in ("train", "valid"):
        _, coarse = load_coarse_split(artifact, cfg, split)
        descriptor = build_transition_descriptor(coarse, **artifact["descriptor_cfg"])
        descriptor = normalize_descriptor(descriptor, prior.descriptor_mean, prior.descriptor_std)
        labels = nearest_labels(descriptor, prior.centers)
        counts[split] = torch.bincount(labels, minlength=regimes).tolist()
    reference = prior_dir / "prior_reference_metrics.json"
    require_files([reference])
    metrics = json.loads(reference.read_text())
    if int(metrics["num_regimes"]) != regimes:
        raise ValueError("Reference metrics and prior disagree on regime count")
    sigma2, koopman = prior.sigma2.float(), prior.koopman_k.float()
    row = {"num_regimes": regimes, "ridge": artifact.get("train_config", {}).get("lambda_k_ridge"),
           "train_counts": counts["train"], "valid_counts": counts["valid"],
           "min_train_count": min(counts["train"]), "min_valid_count": min(counts["valid"]),
           "validation_unvisited_regimes": sum(n == 0 for n in counts["valid"]),
           "k_fro_max": float(torch.linalg.matrix_norm(koopman, ord="fro", dim=(-2, -1)).max()),
           "sigma2_mean_per_regime": sigma2.mean(dim=-1).tolist(),
           "sigma2_floor_regime_fraction": float((sigma2.mean(dim=-1) <= 1.0001e-6).float().mean()),
           "dyn_whitened_val": metrics["dyn_whitened_val"]}
    return {"rows": [row], "scope": "supplied fitted prior and training/validation splits"}


def association(manifest):
    """Build absolute SN/LSKP differences, resampling case-horizon pairs only."""
    from . import association as kernels
    from .tube import align_and_project, score_chunk
    records = json.loads(manifest.read_text())["runs"]
    values, fingerprints = {}, {}
    for run in records:
        case, horizon, model = run["case"], int(run["horizon"]), run["model"]
        if model not in ("sn", "lskp"):
            raise ValueError("Association requires sn and lskp entries only")
        key = case, horizon, model
        if key in values:
            raise ValueError("Duplicate association cell: " + str(key))
        prior_path = resolve_path(run["prior"])
        fingerprint = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        pair = case, horizon
        if pair in fingerprints and fingerprints[pair] != fingerprint:
            raise ValueError("SN and LSKP must be scored with the same frozen prior")
        fingerprints[pair] = fingerprint
        artifact = torch.load(prior_path, map_location="cpu")
        scores = []
        for path in _generated(resolve_path(run["generated"]), horizon):
            raw = torch.from_numpy(np.load(path).astype(np.float32, copy=True))
            coarse = align_and_project(raw, artifact, full_sequence=True)
            scores.append(score_chunk(coarse, artifact)[0])
        metrics = json.loads(resolve_path(run["metrics"]).read_text())
        values[key] = float(np.concatenate(scores).mean()), metrics
    rows = []
    for case, horizon in sorted(fingerprints):
        if (case, horizon, "sn") not in values or (case, horizon, "lskp") not in values:
            raise ValueError("Unpaired case-horizon: {} {}".format(case, horizon))
        sn_tube, sn_metrics = values[case, horizon, "sn"]
        lskp_tube, lskp_metrics = values[case, horizon, "lskp"]
        for metric in kernels.PRIMARY:
            tube = lskp_tube - sn_tube
            # The original association uses aggregate horizon readouts, not
            # bootstrap window means (the two are nonlinear, distinct estimates).
            delta = float(lskp_metrics[metric]) - float(sn_metrics[metric])
            if not math.isfinite(tube) or not math.isfinite(delta):
                raise ValueError("Non-finite association pair")
            label = ("both_lower_lskp" if tube < 0 and delta < 0 else
                     "both_higher_lskp" if tube > 0 and delta > 0 else "decoupled")
            rows.append({"case": case, "horizon": horizon, "metric": metric,
                         "delta_tube": tube, "delta_metric": delta, "alignment": label})
    if len(fingerprints) < 3:
        raise ValueError("Association requires at least three case-horizon pairs")
    rng = np.random.default_rng(3407)
    paper_pairs = {(case, horizon) for case, horizons in (
        ("case1", (1920, 2560, 3200, 3840)), ("case2", (256, 384, 512, 768)),
        ("case3", (256, 384, 512, 768, 1024, 1536))) for horizon in horizons}
    return {"rows": rows, "bootstrap": [kernels.bootstrap_summary(rows, m, rng) for m in kernels.PRIMARY],
            "stratified": kernels.stratified_rows(rows), "n_case_horizon_pairs": len(fingerprints),
            "paper_grid_complete": set(fingerprints) == paper_pairs,
            "resamples": 1000, "seed": 3407, "metric_estimator": "aggregate scorecard",
            "scope": "descriptive pair association, not causal or training-seed uncertainty"}


def parser():
    """List numerical tasks without loading checkpoints or initializing CUDA."""
    result = argparse.ArgumentParser(description="Supplementary CPU numerical analyses")
    children = result.add_subparsers(dest="task", required=True)
    for name in ("pca", "descriptor-coherence", "input-gradients", "tube-calibration", "reliability-gate",
                 "prior-health", "amplification", "memorization"):
        child = children.add_parser(name)
        child.add_argument("--case", choices=("case1", "case2", "case3"), required=True)
        child.add_argument("--prior-dir", help="Default: outputs/prior/CASE")
        for split in ("train", "valid", "test"):
            child.add_argument("--" + split, help="Explicit latent NPY; default: data/latents/CASE/sn/SPLIT_data.npy")
        child.add_argument("--generated", help="Parent of seed_*/gen_latents.npy")
        child.add_argument("--out-dir", required=True)
    for name in ("training-seeds", "common-steps", "metric-grid", "tube-association"):
        child = children.add_parser(name)
        child.add_argument("--manifest", required=True)
        child.add_argument("--out-dir", required=True)
    child = children.add_parser("validation-trajectory")
    child.add_argument("--log", required=True)
    child.add_argument("--best-report")
    child.add_argument("--out-dir", required=True)
    child = children.add_parser("coupling-health")
    child.add_argument("--config", required=True)
    child.add_argument("--run-dir", required=True)
    child.add_argument("--out-dir", required=True)
    return result


def main(argv=None):
    """Run one explicit analysis and write new output files after it succeeds."""
    cli = parser()
    args = cli.parse_args(argv)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    if resolve_path(args.out_dir).exists():
        raise FileExistsError("Choose a new --out-dir: " + args.out_dir)
    if args.task in ("training-seeds", "common-steps", "metric-grid"):
        manifest = json.loads(resolve_path(args.manifest).read_text())
        records = [dict(row, metrics=str(resolve_path(row["metrics"]))) for row in manifest["runs"]]
        result = summarize_metrics(records, args.task)
    elif args.task == "tube-association":
        result = association(resolve_path(args.manifest))
    elif args.task == "validation-trajectory":
        from .validation import parse_log, load_best_report
        path = resolve_path(args.log)
        require_files([path] + ([resolve_path(args.best_report)] if args.best_report else []))
        rows, updates, resumes = parse_log(path)
        if not rows:
            raise ValueError("No validation-loss rows in log; no trajectory can be reconstructed")
        result = {"rows": rows, "best_updates": updates, "best_resumes": resumes,
                  "best_report": load_best_report(resolve_path(args.best_report)) if args.best_report else None,
                  "scope": "logged losses/accepted selector events only; no missing checkpoints inferred"}
    elif args.task == "coupling-health":
        from .coupling import summarize_run
        cfg = load_config(args.config)
        directory = resolve_path(args.run_dir)
        require_files([directory / "log.txt"])
        run = {"recipe": str(resolve_path(args.config)), "run_dir": str(directory),
               "target_step": cfg.get("max_train_steps", 0), "run_id": directory.name,
               "case": "user-specified", "role": "candidate", "status": "supplied_log"}
        row = summarize_run(run, ROOT)
        if not row["n_log_blocks"]:
            raise ValueError("No training log blocks found")
        result = {"rows": [row], "scope": "logged lambda-times-raw-loss scale proxy, not an exact effective loss"}
    else:
        prior_dir = resolve_path(args.prior_dir or "outputs/prior/" + args.case)
        splits = _paths(args)
        artifact_path = prior_dir / "lskp_prior.pt"
        require_files([artifact_path])
        if args.task in ("pca", "input-gradients", "descriptor-coherence", "prior-health"):
            required = ("valid",) if args.task == "input-gradients" else (
                ("train", "valid", "test") if args.task == "descriptor-coherence" else ("train", "valid"))
            bound = {s: splits[s] for s in required}
            require_files(bound.values())
            if args.task == "pca":
                from .pca import audit_case
                result = audit_case(args.case, prior_dir, bound)
            elif args.task == "input-gradients":
                from .gradients import audit_case
                result = audit_case(args.case, prior_dir, bound)
            elif args.task == "descriptor-coherence":
                from .coherence import audit_case
                result = audit_case(prior_dir, bound)
            else:
                result = prior_health(prior_dir, bound)
        else:
            if not args.generated:
                cli.error(args.task + " requires --generated")
            generated = resolve_path(args.generated)
            generation_paths = _generated(generated)
            spec = dict(splits, prior=artifact_path, generated=generated)
            if args.task == "tube-calibration":
                from .tube import audit_case
                require_files(splits.values())
                result = {"rows": audit_case(args.case, {k: str(v) for k, v in spec.items()}, 256)}
            elif args.task == "reliability-gate":
                if args.case != "case1":
                    cli.error("The fixed reliability-gate subgroup protocol is Case1 only")
                from .gate import (burst_threshold, collect_chunks, generated_chunks, summarize_split,
                                   cluster_bootstrap, iter_npy_chunks)
                require_files(splits.values())
                artifact = torch.load(artifact_path, map_location="cpu")
                threshold = burst_threshold(artifact)
                rows, boot = [], []
                for split, path in splits.items():
                    arrays = collect_chunks(iter_npy_chunks(str(path), 256), artifact, full_sequence=False)
                    rows.extend(summarize_split(split, *arrays, threshold))
                    for block, seed in ((1, 3407), (32, 3408)):
                        boot.append(cluster_bootstrap(split, *arrays, threshold, rows_per_cluster=block,
                                    resamples=1000, seed=seed, resampling_unit="sequence window" if block == 1
                                    else "contiguous 32-window block"))
                arrays = collect_chunks(generated_chunks(str(generated)), artifact, full_sequence=True)
                rows.extend(summarize_split("generated_lskp", *arrays, threshold))
                result = {"rows": rows, "bootstrap": boot, "confidence_threshold": .15,
                          "burst_confidence_threshold": .25, "stored_burst_proxy_threshold": threshold,
                          "generated_intervals": "excluded; descriptive only", "threshold_sweep": False}
            elif args.task == "amplification":
                from .amplification import summarize_amplification_for_root
                artifact = torch.load(artifact_path, map_location="cpu")
                regimes = artifact["local_stochastic_prior"]["koopman_k"].shape[0]
                if regimes > 1 and np.load(generation_paths[0], mmap_mode="r").shape[0] != 1:
                    raise ValueError("Multi-regime amplification requires one trajectory per seed")
                horizon = int(np.load(generation_paths[0], mmap_mode="r").shape[1])
                result = {"rows": summarize_amplification_for_root(args.case, "supplied", horizon, generated, artifact)}
            else:
                from .memorization import audit_case
                require_files([splits["train"], splits["test"]])
                artifact = torch.load(artifact_path, map_location="cpu")
                spec["window"] = int(artifact["window_size"])
                settings = argparse.Namespace(train_chunk=128, test_sample=256, duplicate_threshold=1e-4)
                rows, meta = audit_case(args.case, spec, settings)
                result = {"rows": rows, "metadata": meta}
        result["case"] = args.case
        result["prior"] = str(artifact_path)
    result["analysis"] = args.task
    write_result(args.out_dir, result)
    return 0
