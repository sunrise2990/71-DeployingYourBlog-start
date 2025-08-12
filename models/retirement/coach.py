# models/retirement/coach.py
from __future__ import annotations
from typing import Dict, List, Any, Optional

# Keep routes.py thin; import here
from models.retirement.retirement_calc import (
    run_mc_with_seed,
    run_monte_carlo_simulation_locked_inputs as _MC,
)

# ---------------------- helpers ----------------------

def _first_depletion_age(series: Optional[List[float]], start_age: int) -> Optional[int]:
    """Return the first age where assets fall <= 0 (None if never)."""
    if not series:
        return None
    for i, v in enumerate(series):
        try:
            if v is not None and float(v) <= 0.0:
                return int(start_age + i)
        except Exception:
            continue
    return None

def _value_at_age(series: Optional[List[float]], start_age: int, age: int) -> Optional[float]:
    if not series:
        return None
    idx = int(age) - int(start_age)
    if 0 <= idx < len(series):
        try:
            return float(series[idx])
        except Exception:
            return None
    return None

def _quick_mc(params: Dict[str, Any], n_sims: int = 400, seed: int = 12345) -> Dict[str, List[float]]:
    """Fast MC for testing patches. Returns percentiles dict."""
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

# ---------------- targeted solver --------------------

def _meets_success_target(pct: Dict[str, List[float]], start_age: int, target_age: int) -> bool:
    """p10 never drops <= 0 through target_age."""
    p10 = pct.get("p10") or []
    fa = _first_depletion_age(p10, start_age)
    return fa is None or fa > int(target_age)

def _meets_assets_target(
    pct: Dict[str, List[float]], start_age: int, target_age: int, target_amount: float, series: str
) -> bool:
    seq = (pct.get(series) or []) if series in ("p10", "p50", "p90") else (pct.get("p50") or [])
    val = _value_at_age(seq, start_age, target_age)
    return (val is not None) and (val >= float(target_amount))

def _apply_goal_patch_if_needed(working: Dict[str, Any], add_pt_income: bool, amount: float = 6000.0, years: int = 3):
    if not add_pt_income:
        return
    ra = int(working["retirement_age"])
    # Ensure we have a goals list to carry later (client merges to liquidations)
    goals = list(working.get("goal_events") or [])
    goals.append({
        "name": "part-time",
        "is_expense": False,
        "amount": float(amount),
        "start_age": ra,
        "recurrence": "years",
        "years": int(years),
        "inflation_linked": True,
        "enabled": True,
    })
    working["goal_events"] = goals

def coach_solve(params: Dict[str, Any], prefs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Targeted solver. Caller supplies:
      prefs = {
        "target": "success" | "assets",
        "target_age": 85,                      # required
        "target_amount": 2000000,              # for assets target
        "series": "p50",                       # p10|p50|p90 (assets target)
        "levers": ["retirement_age","annual_expense","annual_saving","part_time_income"]
      }
    Returns a single concrete 'action' patch (plus a status header).
    """
    start_age = int(params.get("current_age", 50))
    target = (prefs or {}).get("target", "success")
    target_age = int((prefs or {}).get("target_age", params.get("life_expectancy", start_age + 30)))
    series = (prefs or {}).get("series", "p50")
    target_amount = float((prefs or {}).get("target_amount", 0))
    levers = set((prefs or {}).get("levers") or [])

    # Evaluate current plan quickly
    base_pct = _quick_mc(params, n_sims=400)
    ok_now = False
    if target == "success":
        ok_now = _meets_success_target(base_pct, start_age, target_age)
    else:
        ok_now = _meets_assets_target(base_pct, start_age, target_age, target_amount, series)

    suggestions: List[Dict[str, Any]] = []
    status_txt = "Already meets target" if ok_now else "Does not meet target"
    suggestions.append({"type": "status", "title": "Solver status", "detail": status_txt})

    if ok_now:
        return suggestions

    # Greedy/stepwise solver with small caps (keeps it fast + understandable)
    working = dict(params)
    patch: Dict[str, Any] = {}
    added_pt = False
    max_iters = 30

    # step sizes / bounds
    retire_max_delta = 5
    retire_used = 0

    spend_step, spend_cap = 3000.0, 15000.0
    spend_used = 0.0

    save_step, save_cap = 2000.0, 20000.0
    save_used = 0.0

    for _ in range(max_iters):
        pct = _quick_mc(working, n_sims=400)
        if target == "success":
            if _meets_success_target(pct, start_age, target_age):
                break
        else:
            if _meets_assets_target(pct, start_age, target_age, target_amount, series):
                break

        # Try retirement_age +1 first (if allowed)
        if "retirement_age" in levers and retire_used < retire_max_delta and (working["retirement_age"] < target_age):
            working["retirement_age"] = int(working["retirement_age"]) + 1
            retire_used += 1
            patch["retirement_age"] = int(working["retirement_age"])
            continue

        # Then trim spending a bit (if allowed)
        if "annual_expense" in levers and spend_used < spend_cap:
            working["annual_expense"] = max(0.0, float(working["annual_expense"]) - spend_step)
            spend_used += spend_step
            patch["annual_expense"] = float(working["annual_expense"])
            continue

        # Then increase saving (if allowed)
        if "annual_saving" in levers and save_used < save_cap:
            working["annual_saving"] = float(working["annual_saving"]) + save_step
            save_used += save_step
            patch["annual_saving"] = float(working["annual_saving"])
            continue

        # Finally add part-time income (once) if allowed
        if "part_time_income" in levers and not added_pt:
            _apply_goal_patch_if_needed(working, True, amount=6000.0, years=3)
            added_pt = True
            # encode the same goal in the client-friendly patch
            patch.setdefault("goal_events", []).append({
                "name": "part-time",
                "is_expense": False,
                "amount": 6000.0,
                "start_age": int(working["retirement_age"]),
                "recurrence": "years",
                "years": 3,
                "inflation_linked": True
            })
            continue

        # If we ran out of levers/caps, stop
        break

    # Recheck final
    final_pct = _quick_mc(working, n_sims=400)
    solved = False
    if target == "success":
        solved = _meets_success_target(final_pct, start_age, target_age)
    else:
        solved = _meets_assets_target(final_pct, start_age, target_age, target_amount, series)

    # Build the suggestion/action
    if patch:
        why_bits = []
        if "retirement_age" in patch:  why_bits.append(f"retire at {patch['retirement_age']}")
        if "annual_expense" in patch:  why_bits.append(f"spending → ${patch['annual_expense']:,.0f}/yr")
        if "annual_saving" in patch:   why_bits.append(f"saving → ${patch['annual_saving']:,.0f}/yr")
        if patch.get("goal_events"):    why_bits.append("add part-time income 3 yrs")
        goal = "Target reached" if solved else "Closer to target"
        suggestions.append({
            "type": "action",
            "title": f"{goal}: " + ", ".join(why_bits) if why_bits else goal,
            "why": "Solver applied the smallest steps across your chosen levers to hit the target.",
            "patch": patch
        })
    else:
        suggestions.append({
            "type": "status",
            "title": "No feasible change within caps",
            "detail": "Try enabling more levers or widening caps."
        })

    return suggestions

# --------------- legacy/heuristic coach ----------------

def coach_suggestions(params: Dict[str, Any], percentiles: Dict[str, List[float]], prefs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    If prefs['target'] provided → run targeted solver.
    Otherwise return lightweight heuristic ideas (backwards compatible).
    """
    prefs = prefs or {}
    if prefs.get("target"):
        return coach_solve(params, prefs)

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
        # Simple one-offs
        base = dict(params)
        # Delay 1–5 years: first that works
        for d in range(1, 6):
            try_params = dict(base, retirement_age=int(base["retirement_age"]) + d)
            pct_try = _quick_mc(try_params, n_sims=400)
            if _first_depletion_age(pct_try.get("p10") or [], start_age) is None:
                suggestions.append({
                    "type": "action",
                    "title": f"Delay retirement by {d} year(s)",
                    "why": "Raises success probability; worst-case (p10) stays above $0.",
                    "patch": {"retirement_age": int(base["retirement_age"]) + d}
                })
                break

        # Trim spending up to $12k/yr
        step, max_cut = 3000, 12000
        for cut in range(step, max_cut + step, step):
            try_params = dict(params, annual_expense=max(0, float(params["annual_expense"]) - cut))
            pct_try = _quick_mc(try_params, n_sims=400)
            if _first_depletion_age(pct_try.get("p10") or [], start_age) is None:
                suggestions.append({
                    "type": "action",
                    "title": f"Trim spending by ${cut:,.0f}/yr",
                    "why": "Keeps the 10th percentile (p10) above $0 for the full horizon.",
                    "patch": {"annual_expense": float(params["annual_expense"]) - cut}
                })
                break

        # Part-time income
        ra = int(params["retirement_age"])
        suggestions.append({
            "type": "action",
            "title": "Add part-time income $6,000/yr for 3 years post-retirement",
            "why": "Offsets early sequence risk in initial retirement years.",
            "patch": {"goal_events": [{
                "name": "part-time", "is_expense": False, "amount": 6000.0,
                "start_age": ra, "recurrence": "years", "years": 3, "inflation_linked": True
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


