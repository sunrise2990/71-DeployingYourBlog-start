import numpy as np
from datetime import datetime

# 🔹 Deterministic Retirement Projection (spend-first / deposit-end-of-year)
def run_retirement_projection(
    current_age: int,
    retirement_age: int,
    annual_saving: float,
    saving_increase_rate: float,
    current_assets: float,
    return_rate: float,
    return_rate_after: float,
    annual_expense: float,
    cpp_monthly: float,
    cpp_start_age: int,
    cpp_end_age: int,
    asset_liquidations: list[dict],
    inflation_rate: float,
    life_expectancy: int,
    income_tax_rate: float = 0.15,
):
    """
    Yearly roll-forward using spend-first timing:
      • Pre-retirement: (assets [+ liquidation]) * (1 + r_pre) + savings   ← deposits at END of year
      • Retirement:     max(0, assets - withdrawal [+ liquidation]) * (1 + r_post)
    Withdrawal includes an income-tax overlay:
      tax = ((living_exp - cpp) / (1 - tax_rate)) * tax_rate
    CPP inflates from its first-pay year.
    """
    table: list[dict] = []
    assets = float(current_assets)
    start_year = datetime.now().year

    # Helper: sum liquidations at a given age
    def _liq_at(age: int) -> float:
        return float(sum(x.get("amount", 0.0) for x in asset_liquidations if int(x.get("age", -1)) == age))

    for i, age in enumerate(range(int(current_age), int(life_expectancy) + 1)):
        row: dict = {
            "Age": age,
            "Year": str(start_year + i),
            "Retire": "retire" if age == retirement_age else "",
        }

        # --- Inflation-adjusted living expense (from 'current_age')
        years_from_start = max(0, age - current_age)
        living_exp = float(annual_expense) * ((1.0 + float(inflation_rate)) ** years_from_start)
        row["Living_Exp"] = round(living_exp)

        # --- CPP support (inflated from first-pay year only)
        if cpp_start_age <= age <= cpp_end_age:
            years_from_cpp_start = max(0, age - cpp_start_age)
            cpp_support = float(cpp_monthly) * 12.0 * ((1.0 + float(inflation_rate)) ** years_from_cpp_start)
        else:
            cpp_support = 0.0
        row["CPP_Support"] = round(cpp_support) if cpp_support != 0 else None

        # --- Income-tax overlay (retired only; gross-up on living+CPP)
        retired = age >= retirement_age
        if retired and income_tax_rate > 0:
            net_need_for_tax = max(0.0, living_exp - cpp_support)
            income_tax = net_need_for_tax / (1.0 - float(income_tax_rate)) * float(income_tax_rate)
        else:
            income_tax = 0.0
        row["Income_Tax_Payment"] = round(income_tax) if income_tax != 0 else None

        # --- Net retirement expense shown for convenience
        net_expense = (living_exp - cpp_support + income_tax) if retired else 0.0
        row["Living_Exp_Retirement"] = round(net_expense) if retired else None

        # --- Any scheduled liquidation this year (treated as start-of-year inflow)
        liquidation = _liq_at(age)
        row["Asset_Liquidation"] = round(liquidation) if liquidation != 0 else None

        # --- Pick correct return rate
        applied_return_rate = float(return_rate if age < retirement_age else return_rate_after)
        row["Return_Rate"] = applied_return_rate * 100.0

        if not retired:
            # ===== Accumulation year =====
            # Savings applied at the END of the year (deposit timing change)
            saving_factor = (1.0 + float(saving_increase_rate)) ** (age - current_age)
            savings = float(annual_saving) * saving_factor

            # Grow opening balance + start-of-year liquidations (no savings yet)
            base_before_return = assets + liquidation
            base_before_return = max(0.0, base_before_return)

            inv_return = base_before_return * applied_return_rate
            assets = base_before_return + inv_return + savings  # add savings at year-end

            row["Savings"] = round(savings) if savings != 0 else None
            row["Investment_Return"] = round(inv_return)
            row["Asset"] = round(assets)
            row["Asset_Retirement"] = round(assets)
            row["Withdrawal_Rate"] = None

        else:
            # ===== Retirement year =====
            # Withdraw at the BEGINNING of the year, then grow remaining
            withdrawal = max(0.0, net_expense)
            base_before_return = assets - withdrawal + liquidation
            if base_before_return < 0:
                base_before_return = 0.0  # cannot go below zero before return

            inv_return = base_before_return * applied_return_rate
            assets = base_before_return + inv_return

            row["Savings"] = None
            row["Investment_Return"] = round(inv_return)
            row["Asset"] = round(assets)
            row["Asset_Retirement"] = round(assets)
            # Withdrawal rate relative to start-of-year balance if positive
            start_of_year_assets = assets / (1.0 + applied_return_rate) if applied_return_rate > -1.0 else assets
            row["Withdrawal_Rate"] = (
                round((withdrawal / start_of_year_assets) * 100.0, 1) if start_of_year_assets > 0 else None
            )

        table.append(row)

    return {
        "final_assets": round(assets),
        "table": table,
    }


# ========================== Sensitivity Analysis ==============================

def _final_assets_from_params(params: dict) -> float:
    out = run_retirement_projection(**params)
    return float(out.get("final_assets", 0.0))

def sensitivity_analysis(
    baseline_params: dict,
    variables: list[str],
    delta: float = 0.01
) -> dict[str, dict[str, float]]:
    base_assets = _final_assets_from_params(baseline_params)
    sensitivities: dict[str, dict[str, float]] = {}

    cur_age = int(baseline_params.get("current_age", 0))
    max_age = int(baseline_params.get("life_expectancy", cur_age + 60))

    for var in variables:
        if var not in baseline_params or not isinstance(baseline_params[var], (int, float)):
            continue

        if var == "retirement_age":
            ra = float(baseline_params["retirement_age"])
            if ra <= 0:
                sensitivities[var] = {"sensitivity_pct": 0.0, "dollar_impact": 0.0}
                continue

            up_params = dict(baseline_params)
            dn_params = dict(baseline_params)
            up_params["retirement_age"] = int(min(max_age, round(ra + 1)))
            dn_params["retirement_age"] = int(max(cur_age, round(ra - 1)))

            up_assets = _final_assets_from_params(up_params)
            dn_assets = _final_assets_from_params(dn_params)

            slope_per_year = (up_assets - dn_assets) / 2.0
            delta_years = delta * ra
            dollar_impact = slope_per_year * delta_years
            sensitivity_pct = (dollar_impact / base_assets * 100.0) if base_assets else 0.0

            sensitivities[var] = {"sensitivity_pct": sensitivity_pct, "dollar_impact": dollar_impact}
            continue

        # default 1% bump for other scalar variables
        orig = float(baseline_params[var])
        perturbed = dict(baseline_params)
        perturbed[var] = orig * (1.0 + delta)

        new_assets = _final_assets_from_params(perturbed)
        dollar_impact = new_assets - base_assets
        sensitivity_pct = ((new_assets - base_assets) / base_assets * 100.0) if base_assets else 0.0

        sensitivities[var] = {"sensitivity_pct": sensitivity_pct, "dollar_impact": dollar_impact}

    return sensitivities


# ====================== Monte Carlo (spend-first timing) ======================

def run_monte_carlo_simulation_locked_inputs(
    *,
    current_age: int,
    retirement_age: int,
    annual_saving: float,
    saving_increase_rate: float,
    current_assets: float,
    return_mean: float,
    return_mean_after: float,
    return_std: float,
    annual_expense: float,
    inflation_mean: float,
    inflation_std: float,
    cpp_monthly: float,
    cpp_start_age: int,
    cpp_end_age: int,
    asset_liquidations: list,
    life_expectancy: int,
    num_simulations: int = 300,
    income_tax_rate: float = 0.15,
):
    """
    MC version mirroring deterministic timing:
      • Pre-retirement: (assets [+ liquidation]) * (1 + r) + savings        ← deposits at END of year
      • Retirement:     max(0, assets - withdrawal [+ liquidation]) * (1 + r)
    CPP inflates stochastically from its first-pay year using the same
    per-year inflation draws as living expenses (post start age only).
    """
    years = life_expectancy - current_age + 1
    ages = np.arange(current_age, life_expectancy + 1, dtype=int)
    sim_paths = np.zeros((num_simulations, years), dtype=float)

    # Convenience: gather liquidations per age
    liq_map = {}
    for x in asset_liquidations or []:
        try:
            liq_map[int(x.get("age", -1))] = liq_map.get(int(x.get("age", -1)), 0.0) + float(x.get("amount", 0.0))
        except Exception:
            pass

    for s in range(num_simulations):
        assets = float(current_assets)
        cum_infl = 1.0
        cpp_factor = 1.0  # inflates once CPP starts; remains 1.0 before that

        for idx, age in enumerate(ages):
            retired = age >= retirement_age

            # Per-year random draws
            r_mean = float(return_mean_after if retired else return_mean)
            rand_return = np.random.normal(r_mean, float(return_std))
            rand_infl = np.random.normal(float(inflation_mean), float(inflation_std))

            # Update inflation state for living expense
            cum_infl *= (1.0 + rand_infl)
            living_exp = float(annual_expense) * cum_infl

            # CPP support (inflated from start year only using the same inflation draws)
            if cpp_start_age <= age <= cpp_end_age:
                if age > cpp_start_age:
                    cpp_factor *= (1.0 + rand_infl)  # advance only after start year
                cpp_support = float(cpp_monthly) * 12.0 * cpp_factor
            else:
                cpp_support = 0.0

            # Liquidation applied at the start of the year
            liquidation = float(liq_map.get(int(age), 0.0))

            # Income-tax overlay (retired only)
            net_need_for_tax = max(0.0, living_exp - cpp_support)
            income_tax = (net_need_for_tax / (1.0 - income_tax_rate) * income_tax_rate) if retired else 0.0

            if not retired:
                # Deposit savings at END of year
                saving_factor = (1.0 + float(saving_increase_rate)) ** (age - current_age)
                savings = float(annual_saving) * saving_factor

                # Grow opening balance + start-of-year liquidations (no savings yet)
                base_before_return = assets + liquidation
                base_before_return = max(0.0, base_before_return)
                assets = base_before_return * (1.0 + rand_return) + savings
            else:
                # Spend-first
                withdrawal = max(0.0, living_exp - cpp_support + income_tax)
                base_before_return = assets - withdrawal + liquidation
                base_before_return = max(0.0, base_before_return)
                assets = base_before_return * (1.0 + rand_return)

            sim_paths[s, idx] = assets

    p10 = np.percentile(sim_paths, 10, axis=0).round(0)
    p50 = np.percentile(sim_paths, 50, axis=0).round(0)
    p90 = np.percentile(sim_paths, 90, axis=0).round(0)

    probs = _compute_depletion_probabilities(sim_paths, current_age, [75, 85, 90])

    return {
        "ages": ages.tolist(),
        "sim_paths": sim_paths,
        "percentiles": {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()},
        "depletion_probs": probs,
    }


# 🔸 Track % of simulations depleted before checkpoints
def _compute_depletion_probabilities(sim_paths: np.ndarray, start_age: int, checkpoints: list[int]):
    n_sims, n_years = sim_paths.shape
    probs: dict[int, float] = {}
    ever_zero = (sim_paths == 0).any(axis=1).mean()

    for cp_age in checkpoints:
        idx = cp_age - start_age
        if idx < 0:
            probs[cp_age] = 0.0
            continue
        if idx >= n_years:
            idx = n_years - 1
        depleted = (sim_paths[:, : idx + 1] == 0).any(axis=1).mean()
        probs[cp_age] = float(depleted)

    probs["ever"] = float(ever_zero)
    return probs


# ==== Live-WhatIf: RNG seeding utilities (non-invasive) ====
import random as _py_random
import numpy as _np
from contextlib import contextmanager as _ctxmgr

@_ctxmgr
def _mc_seeded(seed: int):
    """Seed python & numpy RNGs, run code, then restore previous state."""
    _py_state = _py_random.getstate()
    _np_state = _np.random.get_state()

    _py_random.seed(seed)
    _np.random.seed(seed)
    try:
        yield
    finally:
        _py_random.setstate(_py_state)
        _np.random.set_state(_np_state)

def run_mc_with_seed(seed: int, runner, *args, **kwargs):
    """Run existing MC function 'runner' reproducibly without modifying it."""
    with _mc_seeded(seed):
        return runner(*args, **kwargs)


# === APPEND-ONLY: Lite v1 tax-lite + RRIF helpers ============================
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class LiteV1TaxParams:
    start_age: int = 53
    end_age: int = 95
    taxable: float = 0.0
    rrsp: float = 0.0
    tfsa: float = 0.0
    monthly_spend: float = 6000.0
    return_rate: float = 0.05       # annual deterministic growth (toy)
    flat_tax_rate: float = 0.25     # tax only on RRSP withdrawals (toy)
    # Optional: honor a draw order if the route supplies one
    draw_order: Optional[List[str]] = None  # e.g. ["tfsa","taxable","rrsp"]

def _withdraw_in_order(
    need: float,
    taxable: float,
    tfsa: float,
    rrsp: float,
    t: float,
    order: List[str]
):
    """Withdraw according to 'order'. RRSP step grosses-up by flat tax t."""
    use_tax = use_tfsa = use_rrsp_g = tax_rrsp = 0.0
    for bucket in order:
        if need <= 0:
            break
        if bucket == "taxable" and taxable > 0:
            take = min(taxable, need)
            taxable -= take
            use_tax += take
            need -= take
        elif bucket == "tfsa" and tfsa > 0:
            take = min(tfsa, need)
            tfsa -= take
            use_tfsa += take
            need -= take
        elif bucket == "rrsp" and rrsp > 0:
            gross = (need / (1.0 - t)) if t < 0.999 else need
            take_g = min(rrsp, gross)
            rrsp -= take_g
            net = take_g * (1.0 - t)
            tax = take_g - net
            use_rrsp_g += take_g
            tax_rrsp += tax
            need -= net
    return need, taxable, tfsa, rrsp, use_tax, use_tfsa, use_rrsp_g, tax_rrsp

def run_lite_tax_det_v1(p: LiteV1TaxParams) -> Dict[str, Any]:
    """Toy deterministic engine with configurable account order & flat RRSP tax."""
    years = list(range(int(p.start_age), int(p.end_age) + 1))
    annual_spend = float(p.monthly_spend) * 12.0
    t = max(0.0, min(0.90, float(p.flat_tax_rate)))

    taxable = float(p.taxable)
    rrsp    = float(p.rrsp)
    tfsa    = float(p.tfsa)

    # Default order if none supplied
    order = [b.lower() for b in (p.draw_order or ["taxable", "tfsa", "rrsp"])]

    rows: List[Dict[str, Any]] = []
    earliest_depletion_age: Optional[int] = None
    total_taxes = 0.0

    for age in years:
        # grow accounts
        g = 1.0 + float(p.return_rate)
        taxable *= g; rrsp *= g; tfsa *= g

        need = annual_spend

        # withdraw according to order
        need, taxable, tfsa, rrsp, use_tax, use_tfsa, use_rrsp_g, tax_rrsp = _withdraw_in_order(
            need, taxable, tfsa, rrsp, t, order
        )
        total_taxes += tax_rrsp

        shortfall = need if need > 0 else None
        if shortfall and earliest_depletion_age is None:
            earliest_depletion_age = age

        total = max(0.0, taxable) + max(0.0, tfsa) + max(0.0, rrsp)
        rows.append({
            "Age": age,
            "From_Taxable": use_tax,
            "From_TFSA": use_tfsa,
            "From_RRSP_Gross": use_rrsp_g,
            "Tax_On_RRSP": tax_rrsp,
            "Shortfall": shortfall,
            "End_Taxable": taxable,
            "End_TFSA": tfsa,
            "End_RRSP": rrsp,
            "End_Total": total,
        })

    return {
        "years": years,
        "annual_spend": annual_spend,
        "flat_tax_rate": t,
        "earliest_depletion_age": earliest_depletion_age,
        "total_taxes": total_taxes,
        "rows": rows[:25],
    }

# ---- RRIF minimums (approx) -------------------------------------------------
@dataclass
class LiteV1TaxRrifParams(LiteV1TaxParams):
    rrif_min: bool = True  # apply approx min from 71+ if True

def _approx_rrif_min_pct(age: int) -> float:
    # simple CRA-like curve for demo: 1 / (90 - age) from 71+
    if age < 71:
        return 0.0
    denom = max(1, 90 - int(age))
    return 1.0 / float(denom)

def run_lite_tax_det_rrif_v1(p: LiteV1TaxRrifParams) -> Dict[str, Any]:
    years = list(range(int(p.start_age), int(p.end_age) + 1))
    annual_spend = float(p.monthly_spend) * 12.0
    t = max(0.0, min(0.90, float(p.flat_tax_rate)))

    taxable = float(p.taxable)
    rrsp    = float(p.rrsp)
    tfsa    = float(p.tfsa)

    order = [b.lower() for b in (p.draw_order or ["taxable", "tfsa", "rrsp"])]

    rows: List[Dict[str, Any]] = []
    earliest: Optional[int] = None
    total_taxes = 0.0

    for age in years:
        g = 1.0 + float(p.return_rate)
        taxable *= g; rrsp *= g; tfsa *= g

        # Forced RRIF min first
        if p.rrif_min:
            pct = _approx_rrif_min_pct(age)
            if pct > 0 and rrsp > 0:
                forced_g = rrsp * pct
                forced_g = min(forced_g, rrsp)
                rrsp -= forced_g
                forced_net = forced_g * (1.0 - t)
                forced_tax = forced_g - forced_net
                taxable += forced_net     # net deposits into taxable (toy)
                total_taxes += forced_tax

        need = annual_spend
        need, taxable, tfsa, rrsp, use_tax, use_tfsa, use_rrsp_g, tax_rrsp = _withdraw_in_order(
            need, taxable, tfsa, rrsp, t, order
        )
        total_taxes += tax_rrsp

        if need > 0 and earliest is None:
            earliest = age

        total = max(0.0, taxable) + max(0.0, tfsa) + max(0.0, rrsp)
        rows.append({
            "Age": age,
            "From_Taxable": use_tax,
            "From_TFSA": use_tfsa,
            "From_RRSP_Gross": use_rrsp_g,
            "Tax_On_RRSP": tax_rrsp,
            "Shortfall": need if need > 0 else None,
            "End_Taxable": taxable,
            "End_TFSA": tfsa,
            "End_RRSP": rrsp,
            "End_Total": total,
        })

    return {
        "years": years,
        "annual_spend": annual_spend,
        "flat_tax_rate": t,
        "rrif_min": bool(p.rrif_min),
        "earliest_depletion_age": earliest,
        "total_taxes": total_taxes,
        "rows": rows[:25],
    }
# === END APPEND-ONLY =========================================================






# === APPEND-ONLY: Tax-Aware Withdrawal Optimizer v1 ==========================
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import math

@dataclass
class OptimizerCfgV1:
    # Ages & timing
    current_age: int
    retirement_age: int
    life_expectancy: int

    # Calendar (for TFSA cap growth; optional)
    start_year: int = field(default_factory=lambda: datetime.now().year)
    tfsa_cagr_after_2025: float = 0.021  # 2.1% growth of cap from 2026+

    # Balances at "today"
    taxable0: float = 0.0
    rrsp0: float = 0.0
    tfsa0: float = 0.0

    # Pre-retirement contributions (targets)
    annual_saving: float = 0.0
    saving_increase_rate: float = 0.0
    pre_tfsa_annual: float = 7000.0
    pre_rrsp_annual: float = 0.0

    # Economics
    annual_expense: float = 0.0
    return_rate: float = 0.05
    return_rate_after: float = 0.05
    inflation_rate: float = 0.02

    # Tax model (lite, bracket mode preferred)
    tax_brackets: Optional[Dict[str, Any]] = None  # {"enabled":True,"bands":[{"limit":..,"rate":..},...]}
    income_tax_rate: float = 0.25  # used only if brackets are OFF (flat)

    # CPP / OAS
    cpp_monthly: float = 0.0
    cpp_start_age: int = 0
    cpp_end_age: int = 0

    oas_monthly: float = 0.0
    oas_start_age: int = 0
    oas_threshold: float = 0.0
    oas_clawback_rate: float = 0.0
    oas_index: bool = True  # index OAS with inflation from start age

    rrif_start_age: int = 0

    # Taxable friction on growth only (already in your UI)
    taxable_drag: float = 0.0

    # Allow UI payload key for lump-sum cash-ins
    asset_liquidations: List[Dict[str, Any]] = field(default_factory=list)

    # Optimizer knobs
    tfsa_penalty_rate: float = 0.01   # tiny "option value" penalty so TFSA isn't always first
    step_net: float = 500.0           # net dollar step for greedy solver


# ---------- helpers (mirror your JS semantics) ----------
def _rrif_min_pct_exact(age: int) -> float:
    t = {71:0.0528,72:0.0540,73:0.0553,74:0.0567,75:0.0582,76:0.0598,77:0.0617,78:0.0636,79:0.0658,
         80:0.0682,81:0.0708,82:0.0738,83:0.0771,84:0.0808,85:0.0851,86:0.0899,87:0.0955,88:0.1021,
         89:0.1099,90:0.1192,91:0.1306,92:0.1449,93:0.1634,94:0.1879}
    if age >= 95: return 0.20
    if age >= 71: return float(t.get(int(age), 0.0))
    return 1.0 / max(1, 90 - int(age))

def _tax_from_brackets(income: float, br: Optional[Dict[str, Any]]) -> float:
    if not br or not br.get("bands"): return 0.30 * max(0.0, income)  # fallback
    y = max(0.0, float(income))
    tax = 0.0
    prev = 0.0
    bands = br["bands"]
    for b in bands:
        limit = float(b.get("limit", math.inf))
        rate = max(0.0, float(b.get("rate", 0.0)))
        upto = min(y, limit)
        if upto > prev:
            tax += (upto - prev) * rate
        if y <= limit:
            return tax
        prev = limit
    # if income beyond last finite limit
    last_rate = float(bands[-1].get("rate", 0.0))
    return tax + max(0.0, y - prev) * last_rate

def _incremental_tax(delta: float, base: float, br: Optional[Dict[str, Any]], flat_rate: float) -> float:
    d = max(0.0, float(delta))
    b = max(0.0, float(base))
    if br and (br.get("enabled") or br.get("bands")):
        return max(0.0, _tax_from_brackets(b + d, br) - _tax_from_brackets(b, br))
    # flat mode
    return d * max(0.0, min(0.90, float(flat_rate)))

def _oas_gross_at_age(cfg: OptimizerCfgV1, age: int) -> float:
    if not cfg.oas_monthly or not cfg.oas_start_age or age < cfg.oas_start_age: return 0.0
    years = max(0, age - int(cfg.oas_start_age))
    factor = (1.0 + cfg.inflation_rate) ** years if cfg.oas_index else 1.0
    return float(cfg.oas_monthly) * 12.0 * factor

def _oas_claw(income: float, cfg: OptimizerCfgV1, age: int) -> float:
    if cfg.oas_threshold <= 0 or cfg.oas_clawback_rate <= 0: return 0.0
    oas_g = _oas_gross_at_age(cfg, age)
    claw = max(0.0, float(income) - float(cfg.oas_threshold)) * float(cfg.oas_clawback_rate)
    return min(claw, oas_g)

def _living_expense(cfg: OptimizerCfgV1, age: int) -> float:
    n = max(0, int(age) - int(cfg.current_age))
    return float(cfg.annual_expense) * ((1.0 + float(cfg.inflation_rate)) ** n)

def _cpp_at_age(cfg: OptimizerCfgV1, age: int) -> float:
    if not cfg.cpp_monthly or age < cfg.cpp_start_age or age > cfg.cpp_end_age: return 0.0
    n = max(0, int(age) - int(cfg.cpp_start_age))
    return float(cfg.cpp_monthly) * 12.0 * ((1.0 + float(cfg.inflation_rate)) ** n)

def _tfsa_cap_for_year(year: int, cfg: OptimizerCfgV1) -> float:
    base = max(0.0, float(cfg.pre_tfsa_annual))
    if year <= 2025: return base
    yrs = max(0, int(year) - 2025)
    return base * ((1.0 + float(cfg.tfsa_cagr_after_2025)) ** yrs)

def _liq_map(liqs: List[Dict[str, Any]]) -> Dict[int, float]:
    m: Dict[int, float] = {}
    for x in liqs or []:
        try:
            a = int(x.get("age", -1)); amt = float(x.get("amount", 0.0))
            if a >= 0 and amt:
                m[a] = m.get(a, 0.0) + amt
        except Exception:
            pass
    return m

def _preroll_to_retirement_py(cfg: OptimizerCfgV1, liqs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mirror your JS pre-roll, including TFSA->RRSP->Taxable split and taxable drag on growth."""
    age = int(cfg.current_age)
    r_age = int(cfg.retirement_age)
    taxable = float(cfg.taxable0)
    rrsp = float(cfg.rrsp0)
    tfsa = float(cfg.tfsa0)

    lm = _liq_map(liqs)
    d = max(0.0, min(1.0, float(cfg.taxable_drag)))
    gN = 1.0 + float(cfg.return_rate)
    gT = 1.0 + float(cfg.return_rate) * (1.0 - d)

    save = float(cfg.annual_saving)
    while age < r_age:
        # grow start-of-year
        taxable *= gT
        rrsp *= gN
        tfsa *= gN

        # contribution split for this calendar year
        year = int(cfg.start_year) + (age - int(cfg.current_age))
        tfsa_cap = _tfsa_cap_for_year(year, cfg)
        rrsp_cap = max(0.0, float(cfg.pre_rrsp_annual))
        to_tfsa = min(save, tfsa_cap)
        after_tfsa = max(0.0, save - to_tfsa)
        to_rrsp = min(after_tfsa, rrsp_cap)
        to_tax = max(0.0, after_tfsa - to_rrsp)

        tfsa += to_tfsa
        rrsp += to_rrsp
        taxable += to_tax

        # one-off liquidation
        if age in lm:
            taxable += lm[age]

        # next year's savings growth
        save *= (1.0 + float(cfg.saving_increase_rate))
        age += 1

    return {"taxable": taxable, "rrsp": rrsp, "tfsa": tfsa}

# ---------- core: per-year greedy optimizer ----------
def _optimize_year(need_net: float,
                   balances: Dict[str, float],
                   age: int,
                   cfg: OptimizerCfgV1,
                   base_income_prefix: float) -> Dict[str, Any]:
    """
    Choose RRSP/Taxable/TFSA in small steps to meet need_net while minimizing (tax + OAS clawback).
    - Taxes on RRSP spending are *not* funded from assets in bracket mode (matches your table).
    - Tax on taxable spend is ignored (your current bracket mode ignores it); but taxable draws
      do increase 'income' for OAS purposes (matches your current JS).
    """
    step_net = max(50.0, float(cfg.step_net))
    br = cfg.tax_brackets if (cfg.tax_brackets and (cfg.tax_brackets.get("enabled") or cfg.tax_brackets.get("bands"))) else None
    flat = float(cfg.income_tax_rate)

    t = float(balances.get("taxable", 0.0))
    f = float(balances.get("tfsa", 0.0))
    r = float(balances.get("rrsp", 0.0))

    # results
    draw_tax = draw_tfsa = draw_rrsp_g = 0.0
    tax_rrsp_total = 0.0

    # any forced RRIF minimum first (deposit net to taxable; tax on RRIF min computed via brackets/flat)
    rrif_min_pct = _rrif_min_pct_exact(age) if (cfg.rrif_start_age and age >= int(cfg.rrif_start_age)) else 0.0
    if rrif_min_pct > 0 and r > 0:
        must_gross = r * rrif_min_pct
        base_before = base_income_prefix  # CPP+OAS+previous draws (none yet)
        extra_tax = _incremental_tax(must_gross, base_before, br, flat)
        extra_oas = _oas_claw(base_before + must_gross, cfg, age) - _oas_claw(base_before, cfg, age)
        # net from forced min goes to taxable
        net = must_gross - extra_tax - max(0.0, extra_oas)
        net = max(0.0, net)
        r -= must_gross
        t += net
        tax_rrsp_total += extra_tax
        # forced RRIF income also raises the 'income prefix' for subsequent choices
        base_income_prefix += must_gross
        # NOTE: OAS clawback from forced min is shown in per-year OAS section below

    need = max(0.0, float(need_net))

    while need > 1e-6 and (t + f + r) > 0:
        # base income for marginal calculations = CPP + OAS + prior taxable/RRSP draws this year
        base_inc = base_income_prefix + draw_tax + draw_rrsp_g

        # candidate: RRSP small gross chunk that nets approx 'step_net' (cap by balance)
        # do a small gross try; if rate is high it won't net step_net but that's ok
        rrsp_g_try = min(r, max(step_net / (1.0 - max(0.0, flat if not br else 0.0)), 500.0))
        tax_rrsp_inc = _incremental_tax(rrsp_g_try, base_inc, br, flat)
        oas_inc_rrsp = _oas_claw(base_inc + rrsp_g_try, cfg, age) - _oas_claw(base_inc, cfg, age)
        rrsp_net = max(0.0, rrsp_g_try - tax_rrsp_inc - max(0.0, oas_inc_rrsp))
        rrsp_cost_per_net = (tax_rrsp_inc + max(0.0, oas_inc_rrsp)) / (rrsp_net + 1e-9) if rrsp_net > 0 else float("inf")

        # candidate: Taxable step == step_net net, but OAS may rise
        tax_g_try = min(t, step_net)
        oas_inc_tax = _oas_claw(base_inc + tax_g_try, cfg, age) - _oas_claw(base_inc, cfg, age)
        tax_net = tax_g_try  # no direct tax modeled on taxable spend in your table
        tax_cost_per_net = (max(0.0, oas_inc_tax)) / (tax_net + 1e-9) if tax_net > 0 else float("inf")

        # candidate: TFSA step
        tfsa_net = min(f, step_net)
        tfsa_cost_per_net = cfg.tfsa_penalty_rate  # tiny penalty to avoid burning TFSA unless helpful

        # pick the cheapest marginal source
        picks = [
            ("rrsp", rrsp_cost_per_net, rrsp_net, rrsp_g_try),
            ("taxable", tax_cost_per_net, tax_net, tax_g_try),
            ("tfsa", tfsa_cost_per_net, tfsa_net, tfsa_net)
        ]
        picks.sort(key=lambda x: x[1])
        src, _, net_take, gross_take = picks[0]

        if net_take <= 1e-9:
            # nothing useful from chosen source; break to avoid infinite loop
            break

        if src == "rrsp":
            # commit RRSP chunk
            r -= gross_take
            draw_rrsp_g += gross_take
            tax_rrsp_total += tax_rrsp_inc
            need -= net_take
            base_income_prefix += gross_take
        elif src == "taxable":
            t -= gross_take
            draw_tax += gross_take
            need -= net_take
            base_income_prefix += gross_take
        else:
            # TFSA
            f -= gross_take
            draw_tfsa += gross_take
            need -= net_take
            # base income unchanged

    # OAS bookkeeping for display
    cpp_val = _cpp_at_age(cfg, age)
    oas_g = _oas_gross_at_age(cfg, age)
    income_for_oas = base_income_prefix  # includes CPP+OAS+draws (we already added CPP+OAS via prefix upstream)
    oas_claw = _oas_claw(income_for_oas, cfg, age)
    oas_net = max(0.0, oas_g - oas_claw)

    # end-of-year growth
    d = max(0.0, min(1.0, float(cfg.taxable_drag)))
    gN = 1.0 + float(cfg.return_rate_after)
    gT = 1.0 + float(cfg.return_rate_after) * (1.0 - d)

    # (overlay tax: in bracket mode we do NOT fund RRSP tax from assets; parity with your table)
    # Forced RRIF min tax already counted in tax_rrsp_total; in JS you optionally "overlay" that from taxable when drag==0.
    # Here we simply report tax_rrsp_total; front-end can choose where to show it.

    # liquidations at this age happen after withdrawals, before growth
    # (handled in the driver which knows liq_map)
    return {
        "Age": age,
        "From_Taxable": round(draw_tax),
        "From_TFSA": round(draw_tfsa),
        "From_RRSP_Gross": round(draw_rrsp_g),
        "Tax_On_RRSP": round(tax_rrsp_total),
        "OAS_Gross": round(oas_g),
        "OAS_Clawback": round(oas_claw),
        "OAS_Net": round(oas_net),
        "EndBalancesBeforeGrowth": {"taxable": t, "tfsa": f, "rrsp": r}
    }

def optimize_withdrawals_v1(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone runner:
      - Pre-rolls to retirement using the same TFSA→RRSP→Taxable split and taxable-drag semantics
      - Runs greedy per-year optimizer with RRIF/OAS/brackets awareness
      - Returns rows that mirror your table fields, plus End_* after growth
    """
    cfg = OptimizerCfgV1(**cfg_dict)

    # Prefer the dataclass field; accept legacy misspelling as fallback
    liqs: List[Dict[str, Any]] = (
        cfg.asset_liquidations
        or cfg_dict.get("asset_liquidizations")
        or []
    )

    # pre-roll to retirement (compute balances at retirement age start)
    pre = _preroll_to_retirement_py(cfg, liqs)
    taxable = pre["taxable"]; rrsp = pre["rrsp"]; tfsa = pre["tfsa"]

    rows: List[Dict[str, Any]] = []
    lm = _liq_map(liqs)

    for age in range(int(cfg.retirement_age), int(cfg.life_expectancy) + 1):
        expense = _living_expense(cfg, age)
        cpp_val = _cpp_at_age(cfg, age)
        oas_g = _oas_gross_at_age(cfg, age)

        # baseline (CPP + OAS gross) counted in base_income_prefix for OAS/tax marginal math
        base_income_prefix = (cpp_val or 0.0) + (oas_g or 0.0)

        need_net = max(0.0, expense - cpp_val - oas_g)

        # optimize this year
        result = _optimize_year(
            need_net=need_net,
            balances={"taxable": taxable, "tfsa": tfsa, "rrsp": rrsp},
            age=age,
            cfg=cfg,
            base_income_prefix=base_income_prefix
        )

        # apply one-off liquidation before growth
        EB = result["EndBalancesBeforeGrowth"]
        t = EB["taxable"]; f = EB["tfsa"]; r = EB["rrsp"]
        if age in lm: t += lm[age]

        # grow to end-of-year (taxable drag only on Taxable)
        d = max(0.0, min(1.0, float(cfg.taxable_drag)))
        gN = 1.0 + float(cfg.return_rate_after)
        gT = 1.0 + float(cfg.return_rate_after) * (1.0 - d)
        t0 = t
        t = t0 * gT; f *= gN; r *= gN
        end_total = t + f + r

        row = {
            "Age": age,
            "From_Taxable": result["From_Taxable"],
            "From_TFSA": result["From_TFSA"],
            "From_RRSP_Gross": result["From_RRSP_Gross"],
            "RRIF_Min_Pct": (f"{_rrif_min_pct_exact(age)*100:.2f}%" if cfg.rrif_start_age and age >= int(cfg.rrif_start_age) else "0.00%"),
            "OAS_Gross": result["OAS_Gross"],
            "OAS_Clawback": result["OAS_Clawback"],
            "OAS_Net": result["OAS_Net"],
            "Tax_On_Taxable": 0,  # you can overlay RRIF-min tax from taxable in UI if desired
            "Tax_On_RRSP": result["Tax_On_RRSP"],
            "Taxable_Drag": round((t0 * (cfg.return_rate_after) * d) if d > 0 else 0),
            "End_Taxable": round(t),
            "End_TFSA": round(f),
            "End_RRSP": round(r),
            "End_Total": round(end_total),
        }
        rows.append(row)

        # carry balances forward
        taxable, tfsa, rrsp = t, f, r

    earliest = None
    for r in rows:
        if float(r["End_Total"]) <= 0:
            earliest = int(r["Age"]); break

    return {
        "ok": True,
        "rows": rows,
        "earliest_depletion_age": earliest,
        "notes": "v1 greedy optimizer; RRSP-tax via brackets/flat, OAS clawback aware; TFSA tiny penalty to preserve option value."
    }
# === END APPEND-ONLY: Tax-Aware Withdrawal Optimizer v1 ======================







