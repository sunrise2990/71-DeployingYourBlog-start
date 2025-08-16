# models/retirement/coach.py
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple

from models.retirement.retirement_calc import (
    run_mc_with_seed,
    run_monte_carlo_simulation_locked_inputs as _MC,
    run_retirement_projection as _DET,
)

# Mirror routes.py goal handling so coach evaluations honor goal events
from models.retirement.goals import expand_goals_to_per_age, goals_to_liquidations_adapter


# ---------- small utilities ----------

def _first_depletion_age(series: Optional[List[float]], start_age: int) -> Optional[int]:
    """Return first age where series <= 0, or None if never."""
    if not series:
        return None
    for i, v in enumerate(series):
        try:
            if v is not None and float(v) <= 0.0:
                return int(start_age + i)
        except Exception:
            continue
    return None


def _seed_from_params(params: Dict[str, Any], fallback: int = 12345) -> int:
    """
    Try to use the same MC seed as live_update for exact matching with the graph.
    routes.py stores it server-side; if it was forwarded here, honor it.
    """
    for k in ("mc_seed", "_mc_seed", "__seed", "seed"):
        if k in params:
            try:
                return int(params[k])
            except Exception:
                pass
    return fallback


def _coalesce_liqs(liqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coalesce liquidations by age (same behavior as routes.py)."""
    if not liqs:
        return []
    liqs = sorted(liqs, key=lambda r: int(r.get("age", 0)))
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(liqs):
        age_i = int(liqs[i]["age"])
        amt = 0.0
        while i < len(liqs) and int(liqs[i]["age"]) == age_i:
            amt += float(liqs[i]["amount"])
            i += 1
        amt = round(amt, 2)
        if abs(amt) >= 0.005:
            out.append({"age": age_i, "amount": amt})
    return out


def _merge_goals_into_liqs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand goal_events -> per-age -> liquidations and coalesce.
    Keeps coach evaluations consistent with server-side live_update.
    """
    base_liqs = list(params.get("asset_liquidations") or [])
    goals = list(params.get("goal_events") or [])
    if not goals:
        return base_liqs
    try:
        per_age = expand_goals_to_per_age(
            current_age=int(params["current_age"]),
            life_expectancy=int(params["life_expectancy"]),
            inflation_rate=float(params.get("inflation_rate", 0.0)),
            goals=goals,
        )
        merged = goals_to_liquidations_adapter(current_liqs=base_liqs, per_age=per_age)
        return _coalesce_liqs(merged)
    except Exception:
        # Never break coach on goal merge errors; fall back to whatever we had.
        return base_liqs


def _quick_mc(
    params: Dict[str, Any],
    n_sims: int = 400,
    seed: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Fast MC percentiles honoring goal events/liquidations.
    IMPORTANT: If num_simulations is present in params, we use it so the solver
    matches the What-If graph (same #paths ⇒ same p-curves, given same seed).
    """
    liqs = _merge_goals_into_liqs(params)
    n = int(params.get("num_simulations", n_sims))
    sd = _seed_from_params(params) if seed is None else int(seed)

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
        asset_liquidations=liqs,
        life_expectancy=int(params["life_expectancy"]),
        num_simulations=n,
        income_tax_rate=float(params["income_tax_rate"]),
    )
    out = run_mc_with_seed(sd, _MC, **mc_params)
    return out.get("percentiles", {})  # {p10:[], p50:[], p90:[]}


def _quick_det(params: Dict[str, Any]) -> List[float]:
    """Deterministic curve honoring merged goals/liqs."""
    det_keys = [
        "current_age", "retirement_age", "annual_saving", "saving_increase_rate",
        "current_assets", "return_rate", "return_rate_after", "annual_expense",
        "cpp_monthly", "cpp_start_age", "cpp_end_age",
        "asset_liquidations", "inflation_rate", "life_expectancy", "income_tax_rate",
    ]
    det_params = {k: params[k] for k in det_keys if k in params}
    det_params["asset_liquidations"] = _merge_goals_into_liqs(params)
    out = _DET(**det_params) or {}
    tbl = out.get("table", []) or []
    return [row.get("Asset") for row in tbl]


def _apply_patch(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if k == "goal_events":
            out["goal_events"] = list(out.get("goal_events") or []) + list(v or [])
        else:
            out[k] = v
    return out


def _curve_for_metric(params: Dict[str, Any], metric: str) -> List[float]:
    metric = (metric or "p50").lower()
    if metric == "det":
        return _quick_det(params)
    # Match graph’s settings (same seed + same num_simulations when provided)
    pct = _quick_mc(params, n_sims=int(params.get("num_simulations", 2000)))
    if metric == "p10":
        return pct.get("p10", []) or []
    if metric == "p90":
        return pct.get("p90", []) or []
    return pct.get("p50", []) or []


def _value_at_age(curve: List[float], start_age: int, age: int) -> float:
    """Clamp index to curve length to avoid '0' readings beyond horizon."""
    if not curve:
        return 0.0
    idx = max(0, min(len(curve) - 1, int(age) - int(start_age)))
    try:
        return float(curve[idx] or 0.0)
    except Exception:
        return 0.0


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
        # Delay retirement (try up to +5y)
        base = dict(params)
        for d in range(1, 6):
            try_params = dict(base, retirement_age=int(base["retirement_age"]) + d)
            pct_try = _quick_mc(try_params, n_sims=int(base.get("num_simulations", 1200)))
            if _first_depletion_age(pct_try.get("p10") or [], start_age) is None:
                suggestions.append({
                    "type": "action",
                    "title": f"Delay retirement by {d} year(s)",
                    "why": "Raises success probability; worst-case (p10) stays above $0.",
                    "patch": {"retirement_age": int(base["retirement_age"]) + d}
                })
                break

        # Trim spending in $50/mo steps up to $1,200/mo (matches UI slider granularity)
        step_mo, max_cut_mo = 50, 1200
        for cut_mo in range(step_mo, max_cut_mo + step_mo, step_mo):
            cut = cut_mo * 12.0
            try_params = dict(params, annual_expense=max(0.0, float(params["annual_expense"]) - cut))
            pct_try = _quick_mc(try_params, n_sims=int(params.get("num_simulations", 1200)))
            if _first_depletion_age(pct_try.get("p10") or [], start_age) is None:
                suggestions.append({
                    "type": "action",
                    "title": f"Trim spending by ${cut_mo:,.0f}/mo",
                    "why": "Keeps the 10th percentile (p10) above $0 for the full horizon.",
                    "patch": {"annual_expense": float(params["annual_expense"]) - cut}
                })
                break
    else:
        # Gentle stress test when plan is healthy
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


# ---------- targeted single-lever solver ----------

def _bounds_default(params: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Base UI-like bounds; retirement age is further clamped to horizon/current age."""
    return {
        "retirement_age": (55.0, 75.0),
        "annual_expense": (0.0, 240000.0),
        "annual_saving":  (0.0, 240000.0),
        "post_ret_return": (0.00, 0.12),  # decimal
    }


def _apply_lever_override(params: Dict[str, Any], lever: str, x: float) -> Dict[str, Any]:
    if lever == "retirement_age":
        return {"retirement_age": int(round(x))}
    if lever == "annual_expense":
        return {"annual_expense": float(x)}
    if lever == "annual_saving":
        return {"annual_saving": float(x)}
    if lever == "post_ret_return":  # decimal
        return {"return_rate_after": float(x), "return_mean_after": float(x)}
    return {}


def _increasing_with_x(lever: str) -> bool:
    # Whether assets at target age increase as x increases
    return lever in ("retirement_age", "annual_saving", "post_ret_return")


def _solve_single_lever(params: Dict[str, Any],
                        metric: str,
                        target_assets: float,
                        target_age: int,
                        lever: str,
                        bounds: Optional[Tuple[float, float]] = None) -> Tuple[float, Dict[str, Any], float]:
    """
    Binary search the given lever to reach target_assets at target_age.
    Returns (x_star, patch, achieved_assets).

    Supported levers: retirement_age | annual_expense | annual_saving | post_ret_return
    """
    lever = lever or "retirement_age"
    all_bounds = _bounds_default(params)
    lo, hi = bounds if bounds else all_bounds.get(lever, (0.0, 1.0))

    # Honor horizon and current age for retirement-age lever
    start_age = int(params["current_age"])
    life = int(params["life_expectancy"])
    if lever == "retirement_age":
        lo = max(lo, float(start_age))
        hi = min(hi, float(max(start_age + 1, life - 1)))  # cannot retire ≥ life expectancy

    # Clamp target age to available horizon so “age 90” on life=84 doesn’t force None
    target_age = int(min(max(target_age, start_age), life))

    inc = _increasing_with_x(lever)

    # Use the same seed & sim count as the graph to avoid post-apply drift
    seed = _seed_from_params(params)
    n_eval = int(params.get("num_simulations", 2000))

    # Simple memoization to speed the search
    cache: Dict[float, float] = {}

    def eval_assets(x: float, sims: Optional[int] = None) -> float:
        key = (x, sims or n_eval)
        if key in cache:
            return cache[key]
        p = _apply_patch(params, _apply_lever_override(params, lever, x))
        if sims is not None:
            p = dict(p, num_simulations=int(sims))
        else:
            p = dict(p, num_simulations=n_eval)
        # Keep horizon at least up to target_age (defensive if caller forgot to align UI)
        p["life_expectancy"] = max(int(p["life_expectancy"]), target_age)
        curve = _curve_for_metric(dict(p, mc_seed=seed), metric)
        val = _value_at_age(curve, start_age, target_age)
        cache[key] = val
        return val

    # Bracket & checks
    f_lo = eval_assets(lo) - target_assets
    f_hi = eval_assets(hi) - target_assets

    feasible = True
    if inc:
        if f_lo >= 0:
            hi = lo
            f_hi = f_lo
        elif f_hi < 0:
            feasible = False
    else:
        if f_hi >= 0:
            lo = hi
            f_lo = f_hi
        elif f_lo < 0:
            feasible = False

    if not feasible:
        # Choose the closer bound; report transparently.
        x_edge = hi if (abs(f_hi) <= abs(f_lo)) else lo
        val = eval_assets(x_edge)
        return x_edge, _apply_lever_override(params, lever, x_edge), val

    # Bisection (tight tolerances, but bounded iters for speed)
    max_iter = 16
    tol_x = 0.25 if lever == "retirement_age" else (0.0005 if lever == "post_ret_return" else 50.0)
    tol_f = 500.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = eval_assets(mid) - target_assets
        if abs(f_mid) < tol_f or abs(hi - lo) < tol_x:
            lo = hi = mid
            break
        if inc:
            if f_mid >= 0:
                hi = mid
            else:
                lo = mid
        else:
            if f_mid >= 0:
                lo = mid
            else:
                hi = mid

    # Final pick at full graph fidelity (already using n_eval and same seed)
    x_star = hi if inc else lo
    achieved = eval_assets(x_star)

    # Patch to apply
    patch = _apply_lever_override(params, lever, x_star)
    return x_star, patch, achieved


# ---------- public entry ----------

def coach_suggestions(params: Dict[str, Any],
                      percentiles: Dict[str, List[float]],
                      prefs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    If prefs contains a 'target' and 'lever', run single-lever targeted solver:
      prefs = {
        "target": {"metric":"det|p10|p50|p90", "assets": <number>, "age": <int>},
        "lever":  "retirement_age|annual_expense|annual_saving|post_ret_return",
        "bounds": {"retirement_age": [55,75], ...}  # optional, per lever
      }
    Otherwise, fall back to the lightweight heuristic suggestions.
    Backwards-compat: old shapes (type='p10_nonneg' / 'final_assets_min', levers list) still work.
    """
    prefs = prefs or {}
    target = prefs.get("target")
    lever = prefs.get("lever")
    bounds_all = prefs.get("bounds") or {}

    # ---- Backwards compatibility with older prefs ----
    if target and "type" in target:
        t = target["type"]
        if t == "p10_nonneg":
            target = {"metric": "p10", "assets": 0.0,
                      "age": int(target.get("age", params.get("life_expectancy", 90)))}
            lever = (lever or (prefs.get("levers", ["retirement_age"])[0]))
        elif t == "final_assets_min":
            target = {"metric": "p50",
                      "assets": float(target.get("amount", 0.0)),
                      "age": int(target.get("age", params.get("life_expectancy", 90)))}
            lever = (lever or (prefs.get("levers", ["retirement_age"])[0]))

    # ---- New targeted solve path ----
    if target and lever:
        metric = (target.get("metric") or "p50").lower()
        assets_goal = float(target.get("assets", 0.0))
        # Clamp requested target age to horizon here as well (defensive)
        life = int(params.get("life_expectancy", 90))
        cur = int(params.get("current_age", 50))
        age_goal = int(min(max(int(target.get("age", life)), cur), life))

        b = bounds_all.get(lever)
        x_star, patch, achieved = _solve_single_lever(params, metric, assets_goal, age_goal,
                                                      lever, tuple(b) if b else None)

        # Display formatting (monthly for spending/saving as requested)
        if lever == "retirement_age":
            lever_name = "Retirement age"
            val_txt = f"{int(round(x_star))}"
        elif lever == "post_ret_return":
            lever_name = "Post-ret return"
            val_txt = f"{(float(x_star)*100):.1f}%"
        elif lever == "annual_expense":
            lever_name = "Spending (monthly)"
            val_txt = f"${int(round(x_star/12.0)):,}"
        else:  # annual_saving
            lever_name = "Savings (monthly)"
            val_txt = f"${int(round(x_star/12.0)):,}"

        met_txt = {"det": "Deterministic", "p10": "MC p10", "p50": "MC Median", "p90": "MC p90"}[metric]
        title = f"{lever_name}: {val_txt} (meets {met_txt} ≥ ${int(round(assets_goal)):,} at age {age_goal})"

        return [
            {"type": "status",
             "title": "Solve result",
             "detail": f"{title}. Achieved ≈ ${int(round(achieved)):,}."},
            {"type": "action",
             "title": f"Apply: {lever_name} → {val_txt}",
             "patch": patch}
        ]

    # ---- Fallback: heuristic suggestions (now shows $/mo where relevant) ----
    sug = _heuristic_suggestions(params, percentiles)
    for s in sug:
        if s.get("type") == "action" and isinstance(s.get("patch"), dict):
            p = s["patch"]
            if "annual_expense" in p:
                mo = int(round(max(0.0, float(p["annual_expense"])) / 12.0))
                s["title"] = s["title"].replace("/yr", "/mo").replace("spending by $", f"spending by ${mo:,}")
            if "annual_saving" in p:
                mo = int(round(max(0.0, float(p["annual_saving"])) / 12.0))
                s["title"] = f"Increase savings by ${mo:,}/mo"
    return sug





