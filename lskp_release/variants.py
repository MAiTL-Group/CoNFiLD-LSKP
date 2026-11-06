"""Small, explicit differences from the per-case SN/LSKP base recipes."""

from copy import deepcopy


BASELINE_STEPS = {
    "case1": {"origin": 1000000, "sn": 360000},
    "case2": {"origin": 355000, "sn": 185000},
    "case3": {"origin": 500000, "sn": 155000},
}


def apply_variant(config, stage, case, model=None, chart=None, training_seed=None):
    """Return a copied recipe with only the documented run differences applied.

    All generated paths are relative. No files are read/written and the base
    mapping is never mutated. Full settings are listed in CONFIGURATIONS.md.
    """
    if case not in BASELINE_STEPS:
        raise ValueError("Named variants require case1, case2, or case3")
    if model is not None and model not in ("origin", "sn", "lskp"):
        raise ValueError("Unknown model: {}".format(model))
    if chart is not None and chart not in ("origin", "sn"):
        raise ValueError("Unknown CNF chart: {}".format(chart))
    if training_seed is not None and (not isinstance(training_seed, int)
                                      or isinstance(training_seed, bool)
                                      or not 0 <= training_seed < 2 ** 32):
        raise ValueError("Training seed must be an integer in [0, 2**32)")
    cfg = deepcopy(config)
    if stage == "cnf":
        if model is not None or training_seed is not None:
            raise ValueError("CNF variants use chart, not model or training_seed")
        if chart == "origin":
            cfg["NF"]["use_sn"] = False
            cfg["save_path"] = "outputs/cnf/{}/origin".format(case)
            if case == "case2":
                cfg["epochs"] = 5701
        return cfg
    if chart is not None:
        raise ValueError("chart is only valid for CNF recipes")
    if stage == "diffusion":
        model = model or "lskp"
        if model != "lskp":
            chart_name = "origin" if model == "origin" else "sn"
            cfg["train_data_path"] = "data/latents/{}/{}/train_data.npy".format(case, chart_name)
            cfg["valid_data_path"] = "data/latents/{}/{}/valid_data.npy".format(case, chart_name)
            cfg["max_train_steps"] = BASELINE_STEPS[case][model]
            cfg["valid_interval"] = 1000 if case == "case1" else 100
            cfg.pop("training_schedule", None)
            cfg["koopman"] = {"enabled": False}
            if case == "case1":
                cfg.pop("valid_t_seed", None)  # The original entry point defaults to 42.
                if model == "origin":
                    cfg["resume_optimizer"] = True
                    cfg["resume_ema"] = True
            else:
                # Retain compatible field defaults with the prior disabled.
                cfg["koopman"].update(checkpoint_path=None, window_size=256,
                                      lambda_k=0.5, t_start_ratio=5)
        suffix = "" if training_seed is None else "_seed{}".format(training_seed)
        run = "outputs/diffusion/{}/{}{}".format(case, model, suffix)
        cfg.update(log_path=run, min_val=run + "/minmax/latent_min.npy",
                   max_val=run + "/minmax/latent_max.npy")
        if training_seed is not None:
            cfg["train_seed"] = training_seed
        return cfg
    if stage == "inference":
        if training_seed is not None:
            raise ValueError("Inference uses rollout seed; training_seed is not an inference option")
        model = model or cfg.get("model", "lskp")
        chart_name = "origin" if model == "origin" else "sn"
        run = "outputs/diffusion/{}/{}".format(case, model)
        cfg.update(model=model, ema_path=run + "/ema_0.9999_best.pt",
                   min_val=run + "/minmax/latent_min.npy", max_val=run + "/minmax/latent_max.npy",
                   cnf_config="configs/cnf/{}.yaml".format(case),
                   cnf_normalizer="outputs/cnf/{}/{}/normalizer_params.pt".format(case, chart_name),
                   save_dir="outputs/samples/{}/{}".format(case, model))
        return cfg
    raise ValueError("Named run variants are not supported for stage {}".format(stage))
