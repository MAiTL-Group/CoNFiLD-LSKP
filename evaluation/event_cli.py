from __future__ import annotations

import argparse
import json

from evaluation.events.cli_defaults import CASE_CHOICES, DEFAULT_RUN_ARGS
from evaluation.events.eval_split import run_eval_split
from evaluation.events.runner import run_motivation_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MotivationEval v2.")
    parser.add_argument("--eval-split", type=str, default=None, help="Path to fixed eval-split JSON.")
    parser.add_argument("--eval-split-out-root", type=str, default=None, help="Optional override for eval-split out_root.")
    parser.add_argument("--case", type=str, default=DEFAULT_RUN_ARGS["case"], choices=CASE_CHOICES)
    parser.add_argument("--label", type=str, default=DEFAULT_RUN_ARGS["label"])
    parser.add_argument("--gen-fields", type=str, default=DEFAULT_RUN_ARGS["gen_fields"])
    parser.add_argument("--gen-latents", type=str, default=DEFAULT_RUN_ARGS["gen_latents"])
    parser.add_argument("--gen-manifest", type=str, default=DEFAULT_RUN_ARGS["gen_manifest"])
    parser.add_argument("--dns-fields", type=str, default=DEFAULT_RUN_ARGS["dns_fields"])
    parser.add_argument("--coords", type=str, default=DEFAULT_RUN_ARGS["coords"])
    parser.add_argument("--out-dir", type=str, default=DEFAULT_RUN_ARGS["out_dir"])
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_RUN_ARGS["cache_dir"])
    parser.add_argument("--sample-count", type=int, default=DEFAULT_RUN_ARGS["sample_count"])
    parser.add_argument("--time-length", type=int, default=DEFAULT_RUN_ARGS["time_length"])
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN_ARGS["seed"])
    parser.add_argument("--protocols", type=str, default=DEFAULT_RUN_ARGS["protocols"])
    parser.add_argument("--horizons", type=str, default=DEFAULT_RUN_ARGS["horizons"])
    parser.add_argument("--protocol-a-policies", type=str, default=DEFAULT_RUN_ARGS["protocol_a_policies"])
    parser.add_argument("--protocol-c-policies", type=str, default=DEFAULT_RUN_ARGS["protocol_c_policies"])
    parser.add_argument("--protocol-a-pair-length", type=int, default=DEFAULT_RUN_ARGS["protocol_a_pair_length"])
    parser.add_argument("--paired-window-stride", type=int, default=DEFAULT_RUN_ARGS["paired_window_stride"])
    parser.add_argument("--worst-window-score", type=str, default=DEFAULT_RUN_ARGS["worst_window_score"], choices=["energy_w1", "drift_rel"])
    parser.add_argument("--early-frac", type=float, default=DEFAULT_RUN_ARGS["early_frac"])
    parser.add_argument("--late-frac", type=float, default=DEFAULT_RUN_ARGS["late_frac"])
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_RUN_ARGS["k_neighbors"])
    parser.add_argument("--pair-count", type=int, default=DEFAULT_RUN_ARGS["pair_count"])
    parser.add_argument("--pair-bins", type=int, default=DEFAULT_RUN_ARGS["pair_bins"])
    parser.add_argument("--spectrum-bins", type=int, default=DEFAULT_RUN_ARGS["spectrum_bins"])
    parser.add_argument("--max-spectrum-frames", type=int, default=DEFAULT_RUN_ARGS["max_spectrum_frames"])
    parser.add_argument("--max-structure-frames", type=int, default=DEFAULT_RUN_ARGS["max_structure_frames"])
    parser.add_argument("--max-distribution-values", type=int, default=DEFAULT_RUN_ARGS["max_distribution_values"])
    parser.add_argument("--max-mmd-values", type=int, default=DEFAULT_RUN_ARGS["max_mmd_values"])
    parser.add_argument("--delta-max", type=int, default=DEFAULT_RUN_ARGS["delta_max"])
    parser.add_argument("--delta-pairs", type=int, default=DEFAULT_RUN_ARGS["delta_pairs"])
    parser.add_argument("--acf-max-lag", type=int, default=DEFAULT_RUN_ARGS["acf_max_lag"])
    parser.add_argument("--tail-low-q", type=float, default=DEFAULT_RUN_ARGS["tail_low_q"])
    parser.add_argument("--tail-high-q", type=float, default=DEFAULT_RUN_ARGS["tail_high_q"])
    parser.add_argument("--feature-bins", type=int, default=DEFAULT_RUN_ARGS["feature_bins"])
    parser.add_argument("--max-feature-frames", type=int, default=DEFAULT_RUN_ARGS["max_feature_frames"])
    parser.add_argument("--precision-recall-k", type=int, default=DEFAULT_RUN_ARGS["precision_recall_k"])
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_RUN_ARGS["cluster_count"])
    parser.add_argument("--derivative-batch-size", type=int, default=DEFAULT_RUN_ARGS["derivative_batch_size"])
    parser.add_argument("--spectrum-batch-size", type=int, default=DEFAULT_RUN_ARGS["spectrum_batch_size"])
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_RUN_ARGS["bootstrap_samples"])
    parser.add_argument("--enable-protocol-e", action="store_true")
    parser.add_argument("--event-calm-q", type=float, default=DEFAULT_RUN_ARGS["event_calm_q"])
    parser.add_argument("--event-burst-q", type=float, default=DEFAULT_RUN_ARGS["event_burst_q"])
    parser.add_argument("--event-window-length", type=int, default=DEFAULT_RUN_ARGS["event_window_length"])
    parser.add_argument("--event-window-stride", type=int, default=DEFAULT_RUN_ARGS["event_window_stride"])
    parser.add_argument(
        "--event-score",
        type=str,
        default=DEFAULT_RUN_ARGS["event_score"],
        choices=["dns_window_p95_enstrophy", "dns_window_mean_enstrophy", "dns_window_max_grad2"],
    )
    parser.add_argument(
        "--event-threshold-source",
        type=str,
        default=DEFAULT_RUN_ARGS["event_threshold_source"],
        choices=["dns_reference"],
    )
    parser.add_argument("--event-min-group-samples", type=int, default=DEFAULT_RUN_ARGS["event_min_group_samples"])
    parser.add_argument("--boundary-index", type=str, default=DEFAULT_RUN_ARGS["boundary_index"])
    parser.add_argument("--boundary-normals", type=str, default=DEFAULT_RUN_ARGS["boundary_normals"])
    parser.add_argument("--grid-height", type=int, default=DEFAULT_RUN_ARGS["grid_height"])
    parser.add_argument("--grid-width", type=int, default=DEFAULT_RUN_ARGS["grid_width"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.eval_split:
        result = run_eval_split(args.eval_split, out_root_override=args.eval_split_out_root)
    else:
        if not args.case or not args.out_dir:
            parser.error("single-run mode requires --case and --out-dir")
        result = run_motivation_v2(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
