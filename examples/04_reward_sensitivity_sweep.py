"""Confirmatory reward-weight sweep + emergence checks.

Analysis plan drafted before the run in docs/PREREGISTRATION_idea1_2026-07-04.md
(committed with the results, not externally registered). Modes:
  smoke : 3 weights x 1 seed, fast sanity.
  cem   : full grid x 8 CEM seeds (the confirmatory run) + ankle/hip usage
          + prereg verdicts.
  torque: write results/reward_sweep_confirmatory_torque.csv (torque-unit ratios; no retraining).
  ppo   : PPO learner cross-check at aesthetic_ap in {0.3, 1.0}, 3 seeds (needs the `rl` extra).

Evaluation uses reset seeds 0-7 (EVAL_SEEDS); the CEM fitness averages over reset seeds 0-4
(cem_train default), so only seeds 5-7 are unseen in training: not a held-out evaluation.

Emergence signals reported (labelled "held-out" in the plan; see PREREGISTRATION section 7):
(H-a) the AP-inclination floor VALUE vs the human expert 2.77 deg (reward drives it toward 0,
so a floor at ~2.77 is emergent, not a target);
(H-b) the ankle-vs-hip usage ratio, reported as normalised-command utilisation (CSV column
ankle_hip_ratio_*) and converted to torque units by 1/(tau_hip/tau_bos) (partly targeted: the
penalised inclination uses a COP that only the ankle shifts, and the lean term penalises hip
command);
(H-c) robustness across a broad weight band.
Run: python examples/04_reward_sensitivity_sweep.py [mode]
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy import stats

from pirouette_dynamics.env import PirouetteEnv
from pirouette_dynamics.reward import RewardWeights
from pirouette_dynamics.train import cem_train, evaluate

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "..", "results")
ML_FIXED = 0.3
EVAL_SEEDS = list(range(8))
LIN_EXPERIENCED_AP = 2.77
LIN_NOVICE_AP = 5.62
MIDPOINT = (LIN_EXPERIENCED_AP + LIN_NOVICE_AP) / 2.0  # 4.195 deg: expert-side threshold
AP_GRID = [0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]


def _mean_abs_commands(env: PirouetteEnv, policy, seeds: list[int]) -> tuple[float, float]:
    """(mean |a[0]|, mean |a[3]|) of the clipped normalised commands over rollouts."""
    ank, hip = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            a = np.clip(np.asarray(policy.act(obs), dtype=float), -1.0, 1.0)
            ank.append(abs(float(a[0])))
            hip.append(abs(float(a[3])))
            obs, _r, term, trunc, _info = env.step(a)
            if term or trunc:
                break
    return float(np.mean(ank)), float(np.mean(hip))


def ankle_hip_command_ratio(env: PirouetteEnv, policy, seeds: list[int]) -> float:
    """Ankle-to-hip COMMAND UTILISATION ratio: mean|a[0]| / mean|a[3]| on clipped, normalised
    commands. NOT a torque ratio: the env scales ankle by tau_bos and hip by tau_hip, so the hip
    channel has `capacity_factor(env)` (~5.4) times more torque per unit command."""
    ma, mh = _mean_abs_commands(env, policy, seeds)
    return ma / mh if mh > 1e-6 else float("inf")


# Backward-compatible name (pre-2026-10-15 callers): returns the command-utilisation ratio.
ankle_hip_usage = ankle_hip_command_ratio


def capacity_factor(env: PirouetteEnv) -> float:
    """Hip-to-ankle torque capacity per unit command, tau_hip / tau_bos, read from the env."""
    return float(env.tau_hip) / float(env.tau_bos)


def command_to_torque_ratio(command_ratio: float, factor: float) -> float:
    """Convert an ankle/hip command-utilisation ratio to torque units: ratio / (tau_hip/tau_bos)."""
    return command_ratio / factor


def ankle_hip_torque_ratio(env: PirouetteEnv, policy, seeds: list[int]) -> float:
    """Ankle-to-hip ratio in torque units: mean|a[0] * tau_bos| / mean|a[3] * tau_hip|."""
    ma, mh = _mean_abs_commands(env, policy, seeds)
    den = mh * float(env.tau_hip)
    return ma * float(env.tau_bos) / den if den > 1e-9 else float("inf")


def write_torque_sidecar(src_csv: str, dst_csv: str, factor: float) -> list[dict]:
    """Convert the saved command-ratio columns to torque units (constant factor per env, so mean and
    CI half-width both scale by 1/factor). The source CSV is left untouched."""
    with open(src_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        m, h = float(r["ankle_hip_ratio_mean"]), float(r["ankle_hip_ratio_ci95"])
        out.append({"aesthetic_ap": r["aesthetic_ap"],
                    "command_ratio_mean": m, "command_ratio_ci95": h,
                    "torque_ratio_mean": command_to_torque_ratio(m, factor),
                    "torque_ratio_ci95": command_to_torque_ratio(h, factor),
                    "capacity_factor": factor, "n_survived": r["n_survived"]})
    with open(dst_csv, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wtr.writeheader()
        wtr.writerows(out)
    return out


def _ci(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    n = len(a)
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    half = float(stats.t.ppf(0.975, n - 1)) * float(a.std(ddof=1)) / np.sqrt(n)
    return m, half


def train_point_cem(ap_w: float, seed: int, iters: int, pop: int):
    w = RewardWeights(aesthetic_ap=ap_w, aesthetic_ml=ML_FIXED)
    res = cem_train(lambda: PirouetteEnv(weights=w), iterations=iters, population=pop, seed=seed)
    st = evaluate(PirouetteEnv(weights=w), res.policy, EVAL_SEEDS)
    ratio = ankle_hip_command_ratio(PirouetteEnv(weights=w), res.policy, EVAL_SEEDS)
    return st, ratio


def run_cem(seeds: list[int], iters: int, pop: int) -> list[dict]:
    print(f"CEM confirmatory: grid={AP_GRID}  seeds={seeds}  CEM({iters}it,{pop}pop)  "
          f"expert {LIN_EXPERIENCED_AP} / midpoint {MIDPOINT:.2f} / novice {LIN_NOVICE_AP} deg")
    print(f"{'aes_ap':>7} | {'AP incl (mean+-95%CI)':>22} | {'ML':>6} | {'revs':>5} | "
          f"{'ank/hip':>7} | surv")
    rows = []
    for ap_w in AP_GRID:
        pts = [train_point_cem(ap_w, s, iters, pop) for s in seeds]
        ap_vals = [p[0].peak_inclination_ap_deg for p in pts]
        ratios = [p[1] for p in pts if np.isfinite(p[1])]
        ap_m, ap_h = _ci(ap_vals)
        ratio_m, ratio_h = _ci(ratios) if ratios else (float("nan"), float("nan"))
        surv = sum(0 if p[0].toppled else 1 for p in pts)
        rows.append({"aesthetic_ap": ap_w, "ap_incl_mean": ap_m, "ap_incl_ci95": ap_h,
                     "ml_incl_mean": float(np.mean([p[0].peak_inclination_ml_deg for p in pts])),
                     "revs_mean": float(np.mean([p[0].revolutions for p in pts])),
                     "ankle_hip_ratio_mean": ratio_m, "ankle_hip_ratio_ci95": ratio_h,
                     "n_survived": surv, "n_seeds": len(seeds)})
        print(f"{ap_w:7.2f} | {ap_m:8.2f} +- {ap_h:5.2f} deg | "
              f"{rows[-1]['ml_incl_mean']:6.2f} | {rows[-1]['revs_mean']:5.2f} | "
              f"{ratio_m:5.2f}+-{ratio_h:4.2f} | {surv}/{len(seeds)}")
    return rows


def _passes_band(r: dict) -> bool:
    return r["ap_incl_mean"] <= MIDPOINT and r["n_survived"] >= 6


def _span(run: list[dict]) -> float:
    lo, hi = run[0]["aesthetic_ap"], run[-1]["aesthetic_ap"]
    return hi / lo if lo > 0 else float("inf")


def widest_contiguous_band(rows: list[dict]) -> list[dict]:
    """Pre-registration section 4 requires a CONTIGUOUS weight band: consecutive swept weights that
    all pass. Return the passing run with the largest hi/lo span ([] if none pass)."""
    best: list[dict] = []
    run: list[dict] = []
    for r in sorted(rows, key=lambda r: r["aesthetic_ap"]) + [None]:
        if r is not None and _passes_band(r):
            run.append(r)
            continue
        if run and (not best or _span(run) > _span(best)):
            best = run
        run = []
    return best


def prereg_verdict(rows: list[dict]) -> None:
    print("\n--- prereg decision rules (docs/PREREGISTRATION_idea1_2026-07-04.md) ---")
    band = widest_contiguous_band(rows)
    if band:
        lo, hi = min(r["aesthetic_ap"] for r in band), max(r["aesthetic_ap"] for r in band)
        span = hi / lo if lo > 0 else float("inf")
        h2 = span >= 3.0
        verdict = "CONFIRMED" if h2 else "NOT confirmed (span<3x)"
        print(f"H2 robustness: expert-side (AP<= {MIDPOINT:.2f}, surv>=6/8) band = "
              f"[{lo}, {hi}] (span {span:.1f}x) -> {verdict}")
        ratios = [(r['ankle_hip_ratio_mean'], r['ankle_hip_ratio_ci95']) for r in band
                  if np.isfinite(r['ankle_hip_ratio_mean'])]
        if ratios:
            worst_lo = min(m - h for m, h in ratios)
            print(f"H1c ankle-dominance (command ratio, not torque): min(ratio-CI) in band = "
                  f"{worst_lo:.2f} -> "
                  f"{'CONFIRMED (>1)' if worst_lo > 1.0 else 'NOT confirmed'}")
        hi_arm = [r["ap_incl_mean"] for r in rows
                  if r["aesthetic_ap"] >= 1.0 and r["n_survived"] >= 6]
        floor_ok = bool(hi_arm) and min(hi_arm) > 1.0
        print(f"Non-baked-in floor: min AP over surviving aes_ap>=1.0 = "
              f"{min(hi_arm) if hi_arm else float('nan'):.2f} -> "
              f"{'CONFIRMED (>1 deg, not ->0)' if floor_ok else 'NOT confirmed'}")
    else:
        print("H2 robustness: NO expert-side surviving band -> NOT confirmed")


def run_ppo(iters_note: str = "") -> None:
    from pirouette_dynamics.ppo import train_ppo
    print("PPO cross-check (learner-robustness): aesthetic_ap in {0.3, 1.0}, "
          "3 seeds, 100k steps each")
    for ap_w in (0.3, 1.0):
        w = RewardWeights(aesthetic_ap=ap_w, aesthetic_ml=ML_FIXED)
        aps = []
        for seed in (0, 1, 2):
            pol = train_ppo(
                lambda w=w: PirouetteEnv(weights=w), total_timesteps=100_000, seed=seed
            )
            st = evaluate(PirouetteEnv(weights=w), pol, EVAL_SEEDS)
            aps.append(st.peak_inclination_ap_deg)
        m, h = _ci(aps)
        ok = 1.0 <= m <= 3.5 if ap_w == 1.0 else True
        print(f"  PPO aes_ap={ap_w}: AP incl {m:.2f} +- {h:.2f} deg "
              f"{'(prereg PPO pass range [1.0,3.5])' if ap_w == 1.0 else ''} "
              f"{'PASS' if (ap_w == 1.0 and ok) else ''}")


def _save(rows: list[dict], tag: str) -> None:
    os.makedirs(_RESULTS, exist_ok=True)
    csv_path = os.path.join(_RESULTS, f"reward_sweep_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"CSV -> {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        aps = [r["aesthetic_ap"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(aps, [r["ap_incl_mean"] for r in rows], yerr=[r["ap_incl_ci95"] for r in rows],
                    marker="o", capsize=3, label="emergent AP inclination (mean, 95% CI)")
        ax.plot(aps, [r["ml_incl_mean"] for r in rows],
                marker="s", linestyle="--", label="emergent ML")
        ax.axhline(LIN_EXPERIENCED_AP, color="green", linestyle=":",
                   label=f"Lin experienced {LIN_EXPERIENCED_AP}")
        ax.axhline(LIN_NOVICE_AP, color="red", linestyle=":", label=f"Lin novice {LIN_NOVICE_AP}")
        ax.set_xlabel("reward AP-inclination weight (aesthetic_ap; aesthetic_ml fixed 0.3)")
        ax.set_ylabel("emergent COM-COP inclination (deg)")
        ax.set_title(
            f"Confirmatory sweep ({rows[0]['n_seeds']} seeds): "
            "expert-like AP-dominant robustness")
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(_RESULTS, f"reward_sweep_{tag}.png")
        fig.savefig(png, dpi=130)
        print(f"PNG -> {png}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cem"
    if mode == "smoke":
        global AP_GRID
        AP_GRID = [0.3, 1.0, 3.0]
        rows = run_cem([0], 12, 32)
        _save(rows, "smoke")
    elif mode == "torque":
        factor = capacity_factor(PirouetteEnv())
        out = write_torque_sidecar(os.path.join(_RESULTS, "reward_sweep_confirmatory.csv"),
                                   os.path.join(_RESULTS, "reward_sweep_confirmatory_torque.csv"),
                                   factor)
        for r in out:
            print(f"{float(r['aesthetic_ap']):5.2f}  cmd {r['command_ratio_mean']:.2f}  "
                  f"torque {r['torque_ratio_mean']:.3f} +- {r['torque_ratio_ci95']:.3f}")
    elif mode == "ppo":
        run_ppo()
    else:
        rows = run_cem(list(range(8)), 25, 48)
        _save(rows, "confirmatory")
        prereg_verdict(rows)


if __name__ == "__main__":
    main()
