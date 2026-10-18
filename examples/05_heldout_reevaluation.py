"""Post-hoc held-out re-evaluation of the confirmatory sweep (added 2026-10-15).

NOT part of the pre-registered analysis plan. The confirmatory sweep (examples/04, mode `cem`)
evaluates every policy on reset seeds 0-7, but the CEM fitness already averages over reset seeds
0-4, so five of the eight evaluation starts were seen in training. This script retrains the same
9 weights x 8 CEM seeds (CEM is seeded, so the policies are the confirmatory ones), evaluates
them on reset seeds the optimiser never saw, and re-applies the section-4 decision rules.
See docs/PREREGISTRATION_idea1_2026-07-04.md section 7 (2026-10-15 post-hoc entry).

Outputs
  results/confirmatory_policies.json : the 72 policy parameter vectors plus training settings,
                                       so later re-evaluations need no retraining.
  results/reward_sweep_heldout.csv   : one row per (eval set, weight): mean topple fraction,
                                       survivors, AP/ML means with 95% t-CIs over the 8 training
                                       seeds, the ankle/hip command ratio, and the set-level
                                       verdicts (constant within an eval set).

Run: python examples/05_heldout_reevaluation.py [--retrain] [--extended] [--workers N]
  default    : load results/confirmatory_policies.json if present, else retrain (~7 min with
               10 workers, ~45 min serial); evaluate on reset seeds 0-7 (for comparison) and
               100-107 (held out, n = 8 like the confirmatory evaluation).
  --retrain  : retrain even if the policy file exists.
  --extended : also evaluate on reset seeds 100-131 (n = 32).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pirouette_dynamics.env import PirouetteEnv
from pirouette_dynamics.reward import RewardWeights
from pirouette_dynamics.train import LinearPolicy, cem_train, evaluate, rollout

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE.parent / "results"
_SPEC = importlib.util.spec_from_file_location("sweep", _HERE / "04_reward_sensitivity_sweep.py")
assert _SPEC is not None and _SPEC.loader is not None
sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep)

CEM_ITERS = 25
CEM_POP = 48
TRAIN_SEEDS = list(range(8))
EVAL_SETS = {
    "eval0_7": list(range(8)),  # the confirmatory evaluation set (resets 0-4 also train)
    "held100_107": list(range(100, 108)),
    "held100_131": list(range(100, 132)),
}
POLICY_FILE = _RESULTS / "confirmatory_policies.json"
CSV_FILE = _RESULTS / "reward_sweep_heldout.csv"


def _weights(ap_w: float) -> RewardWeights:
    return RewardWeights(aesthetic_ap=ap_w, aesthetic_ml=sweep.ML_FIXED)


def train_one(job: tuple[float, int]) -> dict:
    """Retrain one confirmatory policy (same settings as examples/04 mode `cem`)."""
    ap_w, seed = job
    w = _weights(ap_w)
    res = cem_train(lambda: PirouetteEnv(weights=w), iterations=CEM_ITERS, population=CEM_POP,
                    seed=seed)
    return {"aesthetic_ap": ap_w, "seed": seed, "params": res.policy.params.tolist()}


def load_policy(rec: dict) -> LinearPolicy:
    env = PirouetteEnv(weights=_weights(rec["aesthetic_ap"]))
    return LinearPolicy(int(env.observation_space.shape[0]),
                        int(env.action_space.shape[0]),
                        np.asarray(rec["params"], dtype=float))


def evaluate_one(job: tuple[dict, str]) -> dict:
    """Evaluate one saved policy on one reset set (same statistics as examples/04)."""
    rec, set_name = job
    seeds = EVAL_SETS[set_name]
    env = PirouetteEnv(weights=_weights(rec["aesthetic_ap"]))
    pol = load_policy(rec)
    st = evaluate(env, pol, seeds)
    return {"aesthetic_ap": rec["aesthetic_ap"], "seed": rec["seed"], "eval_set": set_name,
            "topple_frac": float(np.mean([rollout(env, pol, s).toppled for s in seeds])),
            "ap": st.peak_inclination_ap_deg, "ml": st.peak_inclination_ml_deg,
            "toppled": st.toppled,
            "ratio": sweep.ankle_hip_command_ratio(env, pol, seeds)}


def summarise(evals: list[dict], set_name: str) -> list[dict]:
    """Per-weight rows in the examples/04 schema, plus mean topple fraction and the ML CI."""
    rows = []
    for ap_w in sorted({e["aesthetic_ap"] for e in evals if e["eval_set"] == set_name}):
        pts = [e for e in evals if e["eval_set"] == set_name and e["aesthetic_ap"] == ap_w]
        ap_m, ap_h = sweep._ci([p["ap"] for p in pts])
        ml_m, ml_h = sweep._ci([p["ml"] for p in pts])
        ratios = [p["ratio"] for p in pts if np.isfinite(p["ratio"])]
        r_m, r_h = sweep._ci(ratios) if ratios else (float("nan"), float("nan"))
        rows.append({"eval_set": set_name, "n_resets": len(EVAL_SETS[set_name]),
                     "aesthetic_ap": ap_w,
                     "topple_frac_mean": float(np.mean([p["topple_frac"] for p in pts])),
                     "n_survived": sum(0 if p["toppled"] else 1 for p in pts),
                     "n_seeds": len(pts),
                     "ap_incl_mean": ap_m, "ap_incl_ci95": ap_h,
                     "ml_incl_mean": ml_m, "ml_incl_ci95": ml_h,
                     "ankle_hip_ratio_mean": r_m, "ankle_hip_ratio_ci95": r_h})
    return rows


def verdicts(rows: list[dict]) -> dict:
    """Section-4 decision rules as values (examples/04 `prereg_verdict` prints the same rules).

    H2: the widest contiguous run of weights with AP mean <= MIDPOINT and >= 6/8 survivors spans
    >= 3x. H1c: every in-band lower CI of the ankle/hip command ratio is > 1. Floor: the lowest
    AP mean over surviving weights >= 1.0 is > 1 deg.
    """
    band = sweep.widest_contiguous_band(rows)
    out: dict = {"band_lo": float("nan"), "band_hi": float("nan"), "band_span": 0.0,
                 "h2_confirmed": False, "h1c_min_lower_ci": float("nan"),
                 "h1c_confirmed": False}
    if band:
        span = sweep._span(band)
        out.update(band_lo=band[0]["aesthetic_ap"], band_hi=band[-1]["aesthetic_ap"],
                   band_span=span, h2_confirmed=span >= 3.0)
        lows = [r["ankle_hip_ratio_mean"] - r["ankle_hip_ratio_ci95"] for r in band
                if np.isfinite(r["ankle_hip_ratio_mean"])]
        if lows:
            out.update(h1c_min_lower_ci=min(lows), h1c_confirmed=min(lows) > 1.0)
    hi_arm = [r["ap_incl_mean"] for r in rows if r["aesthetic_ap"] >= 1.0 and r["n_survived"] >= 6]
    out["floor_min_ap"] = min(hi_arm) if hi_arm else float("nan")
    out["floor_confirmed"] = bool(hi_arm) and min(hi_arm) > 1.0
    return out


def _env_versions() -> dict:
    import gymnasium
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "gymnasium": gymnasium.__version__}


def get_policies(retrain: bool, workers: int) -> list[dict]:
    if POLICY_FILE.exists() and not retrain:
        print(f"loading policies <- {POLICY_FILE}")
        return list(json.loads(POLICY_FILE.read_text(encoding="utf-8"))["policies"])
    jobs = [(w, s) for w in sweep.AP_GRID for s in TRAIN_SEEDS]
    print(f"retraining {len(jobs)} policies: CEM({CEM_ITERS}it,{CEM_POP}pop), {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        policies = list(ex.map(train_one, jobs))
    meta = {"source": "examples/04_reward_sensitivity_sweep.py mode cem, retrained by examples/05",
            "cem_iterations": CEM_ITERS, "cem_population": CEM_POP,
            "cem_fitness_reset_seeds": [0, 1, 2, 3, 4], "aesthetic_ml": sweep.ML_FIXED,
            "tested_with": _env_versions()}
    os.makedirs(_RESULTS, exist_ok=True)
    POLICY_FILE.write_text(json.dumps({"meta": meta, "policies": policies}), encoding="utf-8")
    print(f"JSON -> {POLICY_FILE}")
    return policies


def _print_set(name: str, rows: list[dict], v: dict) -> None:
    print(f"\n=== {name} (resets {EVAL_SETS[name][0]}-{EVAL_SETS[name][-1]}) ===")
    print(f"{'aes_ap':>6} | topple | surv | {'AP mean+-CI':>13} | {'ML mean+-CI':>13} | ank/hip")
    for r in rows:
        print(f"{r['aesthetic_ap']:6.1f} | {r['topple_frac_mean']:6.3f} | "
              f"{r['n_survived']}/{r['n_seeds']}  | {r['ap_incl_mean']:5.2f} +- "
              f"{r['ap_incl_ci95']:4.2f} | {r['ml_incl_mean']:5.2f} +- {r['ml_incl_ci95']:4.2f} | "
              f"{r['ankle_hip_ratio_mean']:4.2f}+-{r['ankle_hip_ratio_ci95']:4.2f}")
    print(f"H2 band [{v['band_lo']}, {v['band_hi']}] span {v['band_span']:.1f}x -> "
          f"{'CONFIRMED' if v['h2_confirmed'] else 'NOT confirmed'}")
    print(f"H1c min in-band lower CI (command ratio) {v['h1c_min_lower_ci']:.2f} -> "
          f"{'CONFIRMED' if v['h1c_confirmed'] else 'NOT confirmed'}")
    print(f"floor min AP over surviving aes_ap>=1.0 {v['floor_min_ap']:.2f} -> "
          f"{'CONFIRMED' if v['floor_confirmed'] else 'NOT confirmed'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc held-out re-evaluation (examples/05).")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()

    policies = get_policies(args.retrain, args.workers)
    sets = ["eval0_7", "held100_107"] + (["held100_131"] if args.extended else [])
    jobs = [(p, s) for s in sets for p in policies]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        evals = list(ex.map(evaluate_one, jobs))

    all_rows = []
    for s in sets:
        rows = summarise(evals, s)
        v = verdicts(rows)
        _print_set(s, rows, v)
        all_rows += [{**r, **v} for r in rows]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    print(f"\nCSV -> {CSV_FILE}")


if __name__ == "__main__":
    main()
