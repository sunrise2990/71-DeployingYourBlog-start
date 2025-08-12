# models/retirement/coach.py
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple

from models.retirement.retirement_calc import (
    run_mc_with_seed,
    run_monte_carlo_simulation_locked_inputs as _MC,
)

# ---------- helpers ----------

def _first_depletion_age(series: Optional[List[float]], start_age: int) -> Optional[int]:
    if not series:
        return None
    for i, v in enumerate(series):
        try:
            if v is not None and float(v) <= 0.0:
                return int(start_age + i)
        except Exception:
            continue
    return None

def _quick_mc(params: Dict[str, Any], n_sims: int = 500, seed: int = 12345) -> Dict[str, List[float]]:
    mc_params = dict(
        current_age=int(params["current_age"]),
        retirement_age=int(params["retirement_age"]),
        annual_saving=float(params["annual_saving"]),
        saving_increase_rate=float(params["saving_increase_rate"]),
        current_assets=float(params["current_assets"]),
        return_mean=float(params.get("return_mean", params["return_rate"])),
        return_mean_after=float(params.get("return_mean_after", params["return_rate_after"])),
        return_std=float(params["return_std"]),
        annual_expense=float(params["annual_expense"]),
        inflation_mean=float(params.get("inflation_mean", params["inflation_rate"])),
        inflation_std=float(params["inflation_std"]),
        cpp_monthly=float(params["cpp_monthly"]),
        cpp_start_age=int(params["cpp_start_age"]),
        cpp_end_age=int(params["cpp_end_age"]),
        asset_liquidations=list(params.get("asset_liquidations") or []),
        life_expectancy=int(params["life_expectancy"]),
        num_simulations=int(params.get("num_simulations", n_sims)),
        income_tax_rate=float(params["income_tax_rate"]),
    )
    out = run_mc_with_seed(seed, _MC, **mc_params)
    return out.get("percentiles", {})

def _p10_gap_to_age(pct: Dict[str, List[float]], start_age: int, target_age: int) -> float:
    """Negative gap means failure (needs improvement). We want gap >= 0."""
    p10 = pct.get("p10") or []
    if not p10:
        return -1e12
    last_i = max(0, min(len(p10) - 1, target_age - start_age))
    floor_val = min(float(v or 0.0) for v in p10[: last_i + 1])
    return floor_val  # >=0 means OK; <0 means shortfall

def _p50_final_gap(pct: Dict[str, List[float]], start_age: int, target_age: int, min_amount: float) -> float:
    """Positive gap means we already meet/exceed the target."""
    p50 = pct.get("p50") or []
    if not p50:
        return -1e12
    i = max(0, min(len(p50) - 1, target_age - start_age))
    return float(p50[i]) - float(min_amount)

def _apply_patch(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if k == "goal_events":
            out["goal_events"] = list(out.get("goal_events") or []) + list(v or [])
        else:
            out[k] = v
    return out

# ---------- heuristic (fallback) ----------

def _heuristic_suggestions(params: Dict[str, Any], percentiles: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    start_age = int(params.get("current_age", 50))
    p10 = percentiles.get("p10") or []
    horizon_ok = _first_depletion_age(p10, start_age) is None

    suggestions: List[Dict[str, Any]] = []
    status = "OK (≥90% success proxy)" if horizon_ok else "At risk (<90%)"
    at_age = None if horizon_ok else _first_depletion_age(p10, start_age)
    suggestions.append({
        "type": "status",
        "title": "Plan health",
        "detail": f"{status}" + (f" — depletion risk near age {at_age}" if at_age else "")
    })

    if not horizon_ok:
        # delay retirement suggestion
        base = dict(params)
        for d in range(1, 6):
            try_params = dict(base, retirement_age=int(base["retirement_age"]) + d)
            pct_try = _quick_mc(try_params, n_sims=500)
            if _first_depletion_age(pct_try.get("p10") or [], start_age) is None:
                suggestions.append({
                    "type": "action",
                    "title": f"Delay retirement by {d} year(s)",
                    "why": "Raises success probability; worst-case (p10) stays above $0.",
                    "patch": {"retirement_age": int(base["retirement_age"]) + d}
                })
                break

        # trim spending
        step, max_cut = 3000, 12000
        for cut in range(step, max_cut + step, step):
            try_params = dict(params, annual_expense=max(0, float(params["annual_expense"]) - cut))
            pct_try = _quick_mc(try_params, n_sims=500)
            if _first_depletion_age(pct_try.get("p10") or [], start_age) is None:
                suggestions.append({
                    "type": "action",
                    "title": f"Trim spending by ${cut:,.0f}/yr",
                    "why": "Keeps the 10th percentile (p10) above $0 for the full horizon.",
                    "patch": {"annual_expense": float(params["annual_expense"]) - cut}
                })
                break

        # part-time inflow
        ra = int(params["retirement_age"])
        suggestions.append({
            "type": "action",
            "title": "Add part-time income $6,000/yr for 3 years post-retirement",
            "why": "Offsets early sequence risk in initial retirement years.",
            "patch": {"goal_events": [{
                "name": "part-time",
                "is_expense": False,
                "amount": 6000.0,
                "start_age": ra,
                "recurrence": "years",
                "years": 3,
                "inflation_linked": True
            }]}
        })
    else:
        suggestions.append({
            "type": "action",
            "title": "Stress test: reduce returns by 1% (pre & post)",
            "why": "Check plan resilience under slightly worse markets.",
            "patch": {
                "return_rate": max(0.0, float(params["return_rate"]) - 0.01),
                "return_rate_after": max(0.0, float(params["return_rate_after"]) - 0.01),
                "return_mean": max(0.0, float(params.get("return_mean", params["return_rate"])) - 0.01),
                "return_mean_after": max(0.0, float(params.get("return_mean_after", params["return_rate_after"])) - 0.01),
            }
        })
    return suggestions

# ---------- targeted solver ----------

def _solve_target(params: Dict[str, Any], target: Dict[str, Any], levers: List[str]) -> List[Dict[str, Any]]:
    """
    Greedy coordinate search over allowed levers to meet a user target.
    Targets:
      - {"type":"p10_nonneg", "age": 85}
      - {"type":"final_assets_min", "age": 84, "amount": 2_000_000}
    Levers may include: "retirement_age", "annual_expense", "annual_saving", "part_time_income".
    """
    start_age = int(params["current_age"])
    ra0 = int(params["retirement_age"])
    allowed = set(levers or [])
    steps: List[Tuple[str, Dict[str, Any], float]] = []  # (desc, patch, score_improvement)

    # base score/gap
    pct0 = _quick_mc(params, n_sims=400)
    if target["type"] == "p10_nonneg":
        gap = _p10_gap_to_age(pct0, start_age, int(target["age"]))  # want >= 0
        goal_met = gap >= 0
    elif target["type"] == "final_assets_min":
        gap = _p50_final_gap(pct0, start_age, int(target["age"]), float(target["amount"]))
        goal_met = gap >= 0
    else:
        return [{"type": "status", "title": "Unknown target", "detail": str(target)}]

    if goal_met:
        return [{
            "type": "status",
            "title": "Target already satisfied",
            "detail": "No changes needed."
        }]

    # search
    patch_total: Dict[str, Any] = {}
    MAX_ITERS = 12
    for _ in range(MAX_ITERS):
        best: Tuple[str, Dict[str, Any], float] | None = None

        # candidate: delay retirement age by +1 (cap +10)
        if "retirement_age" in allowed and int(params["retirement_age"]) < ra0 + 10:
            cand = {"retirement_age": int(params["retirement_age"]) + 1}
            pct = _quick_mc(_apply_patch(params, _apply_patch(patch_total, cand)), n_sims=350)
            new_gap = (_p10_gap_to_age(pct, start_age, int(target["age"]))
                       if target["type"] == "p10_nonneg"
                       else _p50_final_gap(pct, start_age, int(target["age"]), float(target["amount"])))
            imp = (new_gap - gap)
            if best is None or imp > best[2]:
                best = (f"Delay retirement to {cand['retirement_age']}", cand, imp)

        # candidate: cut spending by $1k/yr (cap $15k)
        if "annual_expense" in allowed and float(params["annual_expense"]) > 1000:
            cut = 1000.0
            cand = {"annual_expense": max(0.0, float(params["annual_expense"]) - cut)}
            pct = _quick_mc(_apply_patch(params, _apply_patch(patch_total, cand)), n_sims=350)
            new_gap = (_p10_gap_to_age(pct, start_age, int(target["age"]))
                       if target["type"] == "p10_nonneg"
                       else _p50_final_gap(pct, start_age, int(target["age"]), float(target["amount"])))
            imp = (new_gap - gap)
            if best is None or imp > best[2]:
                best = (f"Trim spending to ${cand['annual_expense']:,.0f}/yr", cand, imp)

        # candidate: add $1k/yr saving (pre-ret only effect, but simple)
        if "annual_saving" in allowed:
            bump = 1000.0
            cand = {"annual_saving": float(params["annual_saving"]) + bump}
            pct = _quick_mc(_apply_patch(params, _apply_patch(patch_total, cand)), n_sims=350)
            new_gap = (_p10_gap_to_age(pct, start_age, int(target["age"]))
                       if target["type"] == "p10_nonneg"
                       else _p50_final_gap(pct, start_age, int(target["age"]), float(target["amount"])))
            imp = (new_gap - gap)
            if best is None or imp > best[2]:
                best = (f"Increase saving to ${cand['annual_saving']:,.0f}/yr", cand, imp)

        # candidate: part-time income $6k/yr for 3 years after retirement
        if "part_time_income" in allowed:
            ra = int((_apply_patch(params, patch_total)).get("retirement_age", ra0))
            cand = {"goal_events": [{
                "name": "part-time",
                "is_expense": False,
                "amount": 6000.0,
                "start_age": ra,
                "recurrence": "years",
                "years": 3,
                "inflation_linked": True
            }]}
            pct = _quick_mc(_apply_patch(params, _apply_patch(patch_total, cand)), n_sims=350)
            new_gap = (_p10_gap_to_age(pct, start_age, int(target["age"]))
                       if target["type"] == "p10_nonneg"
                       else _p50_final_gap(pct, start_age, int(target["age"]), float(target["amount"])))
            imp = (new_gap - gap)
            if best is None or imp > best[2]:
                best = (f"Add part-time income $6k/yr ×3y", cand, imp)

        if best is None or best[2] <= 0:
            break  # no improvement from any lever

        # accept best step
        desc, cand_patch, _ = best
        patch_total = _apply_patch(patch_total, cand_patch)
        steps.append(best)
        # update gap baseline
        pct_after = _quick_mc(_apply_patch(params, patch_total), n_sims=350)
        gap = (_p10_gap_to_age(pct_after, start_age, int(target["age"]))
               if target["type"] == "p10_nonneg"
               else _p50_final_gap(pct_after, start_age, int(target["age"]), float(target["amount"])))
        if (target["type"] == "p10_nonneg" and gap >= 0) or (target["type"] == "final_assets_min" and gap >= 0):
            break

    # build response
    out: List[Dict[str, Any]] = []
    final_pct = _quick_mc(_apply_patch(params, patch_total), n_sims=500)
    final_ok = (_p10_gap_to_age(final_pct, start_age, int(target.get("age", start_age))) >= 0
                if target["type"] == "p10_nonneg"
                else _p50_final_gap(final_pct, start_age, int(target["age"]), float(target["amount"])) >= 0)

    title = "Solution found" if final_ok else "Best effort (not fully met)"
    detail = (f"Applied {len(steps)} step(s)."
              + ("" if final_ok else " You can add more levers or widen bounds."))

    out.append({"type": "status", "title": title, "detail": detail})

    if patch_total:
        out.append({
            "type": "action",
            "title": "Apply full solution",
            "why": "Apply all steps in one click.",
            "patch": patch_total
        })

    # also list step-by-step patches
    for desc, cand_patch, _imp in steps:
        out.append({"type": "action", "title": desc, "patch": cand_patch})

    if not steps:
        out.append({"type": "status", "title": "No improving move with chosen levers", "detail": ""})

    return out

# ---------- public entry ----------

def coach_suggestions(params: Dict[str, Any],
                      percentiles: Dict[str, List[float]],
                      prefs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    If prefs contains a 'target' and optional 'levers', run targeted solver.
    Otherwise, fall back to lightweight heuristic suggestions.
    """
    prefs = prefs or {}
    target = prefs.get("target")
    levers = prefs.get("levers") or []
    if target:
        return _solve_target(params, target, levers)
    return _heuristic_suggestions(params, percentiles)



