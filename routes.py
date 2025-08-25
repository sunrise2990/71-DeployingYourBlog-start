from flask import Blueprint, request, render_template, jsonify, flash, redirect, url_for
import json
from flask_login import login_required, current_user
import numpy as np
from models.retirement.retirement_calc import (
    run_retirement_projection,
    run_monte_carlo_simulation_locked_inputs, sensitivity_analysis
)
from models import db
from models.retirement.retirement_scenario import RetirementScenario  # adjust import path as needed

from copy import deepcopy
from flask import current_app

# -------------------------
# Canonical keys + adapters
# -------------------------
_CANON_KEYS = {
    "current_age", "retirement_age",
    "annual_saving", "saving_increase_rate",
    "current_assets", "return_rate", "return_rate_after",
    "annual_expense",
    "cpp_monthly", "cpp_start_age", "cpp_end_age",
    "asset_liquidations",
    "inflation_rate", "life_expectancy", "income_tax_rate",
    "return_std", "inflation_std",
}

def _to_int(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d

def _to_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d

def to_canonical_inputs(raw: dict) -> dict:
    """
    Accepts legacy or new shapes and returns canonical kwargs for the engine.
    Idempotent: safe to run on already-canonical dicts.
    NOTE: Use this single normalizer everywhere (save, compare, etc.).
    """
    p = deepcopy(raw) if raw else {}

    # ---- legacy -> canonical field names / units ----
    if "lifespan" in p and "life_expectancy" not in p:
        p["life_expectancy"] = p.pop("lifespan")

    if "monthly_living_expense" in p and "annual_expense" not in p:
        p["annual_expense"] = _to_float(p.pop("monthly_living_expense")) * 12.0

    if "cpp_support" in p and "cpp_monthly" not in p:
        p["cpp_monthly"] = p.pop("cpp_support")

    if "cpp_from_age" in p and "cpp_start_age" not in p:
        p["cpp_start_age"] = p.pop("cpp_from_age")
    if "cpp_to_age" in p and "cpp_end_age" not in p:
        p["cpp_end_age"] = p.pop("cpp_to_age")

    # Build list-style liquidations if only discrete slots exist
    if "asset_liquidations" not in p:
        alist = []
        for i in (1, 2, 3):
            amt = _to_float(p.pop(f"asset_liquidation_{i}", 0))
            age = _to_int(p.pop(f"asset_liquidation_age_{i}", 0))
            if amt and age:
                alist.append({"amount": amt, "age": age})
        if alist:
            p["asset_liquidations"] = alist

    # ---- light typing on canonical keys ----
    ints = {"current_age", "retirement_age", "cpp_start_age", "cpp_end_age", "life_expectancy"}
    floats = {
        "annual_saving", "saving_increase_rate", "current_assets",
        "return_rate", "return_rate_after", "annual_expense",
        "inflation_rate", "income_tax_rate", "cpp_monthly",
        "return_std", "inflation_std",
    }
    for k in list(p.keys()):
        if k in ints:
            p[k] = _to_int(p.get(k))
        elif k in floats:
            p[k] = _to_float(p.get(k))

    p.setdefault("asset_liquidations", [])

    # keep only what engine accepts
    return {k: p[k] for k in _CANON_KEYS if k in p}


def canonical_to_form_inputs(canon: dict) -> dict:
    d = dict(canon or {})
    out = {}

    # Lifespan
    if "life_expectancy" in d:
        out["lifespan"] = _to_int(d["life_expectancy"])

    # --- Living expense display: show MONTHLY in the form ---
    if "annual_expense" in d:
        out["monthly_living_expense"] = _to_float(d["annual_expense"]) / 12.0
    elif "monthly_living_expense" in d:  # legacy rows
        out["monthly_living_expense"] = _to_float(d["monthly_living_expense"])

    # --- Savings display: show MONTHLY in the form ---
    if "annual_saving" in d:
        # If we have canonical fields (annual_expense present), treat as canonical annual and divide by 12.
        if "annual_expense" in d:
            out["annual_saving"] = _to_float(d["annual_saving"]) / 12.0
        else:
            # Legacy rows likely stored monthly; pass through.
            out["annual_saving"] = _to_float(d["annual_saving"])

    # CPP (legacy names in the form)
    if "cpp_monthly" in d:
        out["cpp_support"] = _to_float(d["cpp_monthly"])
    if "cpp_start_age" in d:
        out["cpp_from_age"] = _to_int(d["cpp_start_age"])
    if "cpp_end_age" in d:
        out["cpp_to_age"] = _to_int(d["cpp_end_age"])

    # Direct pass-throughs
    for k in [
        "current_age", "retirement_age", "saving_increase_rate", "current_assets",
        "return_rate", "return_rate_after", "inflation_rate", "income_tax_rate"
    ]:
        if k in d:
            out[k] = d[k]

    # Asset liquidations -> 3 slots
    als = d.get("asset_liquidations", [])
    for i in range(3):
        amt = als[i]["amount"] if i < len(als) else 0
        age = als[i]["age"] if i < len(als) else 0
        out[f"asset_liquidation_{i+1}"] = amt
        out[f"asset_liquidation_age_{i+1}"] = age

    # Optional MC fields
    if "return_std" in d:
        out["return_std"] = d["return_std"]
    if "inflation_std" in d:
        out["inflation_std"] = d["inflation_std"]

    return out

# Define main projects blueprint (existing)
projects_bp = Blueprint('projects', __name__, template_folder='templates')

# Budget Reforecast route
@projects_bp.route("/projects/budget-reforecast")
def budget_reforecast():
    return render_template("budget_reforecast.html")

# Leasing pipeline route
@projects_bp.route("/projects/budget-reforecast/leasing")
def leasing_pipeline():
    return render_template("leasing_pipeline.html")

@projects_bp.route("/retirement", methods=["GET", "POST"])
def retirement():
    # Initialize all outputs and defaults
    result = None
    table = []

    # Default‐init chart and MC data to avoid missing‐key errors
    chart_data = {
        "Age": [],
        "Living_Exp_Retirement": [],
        "Asset_Retirement": [],
        "Withdrawal_Rate": []
    }
    monte_carlo_data = {
        "Age": [],
        "Percentile_10": [],
        "Percentile_50": [],
        "Percentile_90": []
    }
    depletion_stats = {"age_75": 0.0, "age_85": 0.0, "age_90": 0.0, "ever": 0.0}

    reset = False
    retirement_age = None

    form_inputs = {}
    sensitivities = {}
    baseline_params = {}

    # Headers for the main projection table
    table_headers = [
        "Age", "Year", "Retire?", "Living Exp.", "CPP / Extra Income", "Income Tax Payment",
        "Living Exp. – Ret.", "Asset Liquidation", "Savings – Before Retire", "Asset",
        "Asset – Retirement", "Investment Return", "Return Rate", "Withdrawal Rate"
    ]

    # Sensitivity table scaffolding
    sensitivity_headers = ["Variable", "Sensitivity (%)", "Dollar Impact ($)"]
    sensitivity_table = []

    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset":
            reset = True
        elif action == "calculate":
            try:
                # ---- helpers ---------------------------------------------------
                def get_form_value(name, caster, default=0):
                    val = request.form.get(name)
                    form_inputs[name] = val
                    try:
                        return caster(val) if val not in (None, "") else default
                    except Exception:
                        return default

                def pct_to_dec(name):
                    return get_form_value(name, float, 0.0) / 100.0

                # ---- gather inputs (UI units) --------------------------------
                current_age            = get_form_value("current_age", int, 50)
                retirement_age         = get_form_value("retirement_age", int, current_age + 10)
                monthly_saving_ui      = get_form_value("annual_saving", float, 0.0)  # UI holds *monthly*
                return_rate            = pct_to_dec("return_rate")        # CAGR pre-ret (decimal)
                return_rate_after      = pct_to_dec("return_rate_after")  # CAGR post-ret (decimal)
                lifespan               = get_form_value("lifespan", int, retirement_age + 25)
                monthly_living_expense = get_form_value("monthly_living_expense", float, 0.0)
                inflation_rate         = pct_to_dec("inflation_rate")
                current_assets         = get_form_value("current_assets", float, 0.0)
                saving_increase_rate   = pct_to_dec("saving_increase_rate")
                income_tax_rate        = pct_to_dec("income_tax_rate")

                cpp_monthly = get_form_value("cpp_support", float, 0.0)
                cpp_from    = get_form_value("cpp_from_age", int, retirement_age)
                cpp_to      = get_form_value("cpp_to_age", int, lifespan)

                return_std    = pct_to_dec("return_std")     # σ as decimal
                inflation_std = pct_to_dec("inflation_std")  # σ for inflation (decimal)

                # collect up to 3 liquidation events
                asset_liquidation = []
                for i in range(1, 4):
                    amt = get_form_value(f"asset_liquidation_{i}", float, 0.0)
                    age = get_form_value(f"asset_liquidation_age_{i}", int, 0)
                    if amt and age:
                        asset_liquidation.append({"amount": float(amt), "age": int(age)})

                # guard horizon
                life_expectancy = max(int(lifespan), int(retirement_age))

                # ---- deterministic baseline (uses CAGR directly) --------------
                baseline_params = {
                    "current_age": int(current_age),
                    "retirement_age": int(retirement_age),
                    "annual_saving": float(monthly_saving_ui) * 12.0,  # UI monthly -> annual
                    "saving_increase_rate": float(saving_increase_rate),
                    "current_assets": float(current_assets),
                    "return_rate": float(return_rate),                   # CAGR
                    "return_rate_after": float(return_rate_after),       # CAGR
                    "annual_expense": float(monthly_living_expense) * 12.0,
                    "cpp_monthly": float(cpp_monthly),
                    "cpp_start_age": int(cpp_from),
                    "cpp_end_age": int(cpp_to),
                    "asset_liquidations": asset_liquidation,
                    "inflation_rate": float(inflation_rate),
                    "life_expectancy": life_expectancy,
                    "income_tax_rate": float(income_tax_rate),
                }

                # ---- sensitivity (deterministic params) -----------------------
                variables = [
                    "current_assets", "return_rate", "return_rate_after",
                    "annual_saving", "annual_expense", "saving_increase_rate",
                    "inflation_rate", "income_tax_rate", "retirement_age",
                ]
                try:
                    sensitivities = sensitivity_analysis(baseline_params, variables, delta=0.01)
                    sensitivity_table = [
                        [
                            var,
                            f"{vals.get('sensitivity_pct', 0.0):.2f}%",
                            f"${vals.get('dollar_impact', 0.0):,.0f}",
                        ]
                        for var, vals in sensitivities.items()
                    ]
                except Exception:
                    sensitivities = {}
                    sensitivity_table = []

                # ---- deterministic projection/table ---------------------------
                output = run_retirement_projection(**baseline_params)
                result = output.get("final_assets")

                # safety: ensure keys exist
                rows = output.get("table", []) or []
                for row in rows:
                    if "Living_Exp_Retirement" not in row or row["Living_Exp_Retirement"] is None:
                        row["Living_Exp_Retirement"] = row.get("Living_Exp", 0)

                table = [[
                    r.get("Age"),
                    r.get("Year"),
                    r.get("Retire"),
                    f"${(r.get('Living_Exp') or 0):,.0f}",
                    f"${(r.get('CPP_Support') or 0):,.0f}" if r.get("CPP_Support") else "",
                    f"${(r.get('Income_Tax_Payment') or 0):,.0f}",
                    f"${(r.get('Living_Exp_Retirement') or 0):,.0f}",
                    f"${(r.get('Asset_Liquidation') or 0):,.0f}" if r.get("Asset_Liquidation") else "",
                    f"${(r.get('Savings') or 0):,.0f}" if r.get("Savings") else "",
                    f"${(r.get('Asset') or 0):,.0f}",
                    f"${(r.get('Asset_Retirement') or 0):,.0f}" if r.get("Asset_Retirement") else "",
                    f"${(r.get('Investment_Return') or 0):,.0f}" if r.get("Investment_Return") is not None else "",
                    f"{(r.get('Return_Rate') or 0):.1f}%" if r.get("Return_Rate") is not None else "",
                    f"{(r.get('Withdrawal_Rate') or 0):.1f}%" if r.get("Withdrawal_Rate") is not None else "",
                ] for r in rows]

                chart_data = {
                    "Age": [r.get("Age") for r in rows],
                    "Living_Exp_Retirement": [r.get("Living_Exp_Retirement") or 0 for r in rows],
                    "Asset_Retirement": [r.get("Asset_Retirement") or 0 for r in rows],
                    "Withdrawal_Rate": [
                        ((r.get("Withdrawal_Rate") or 0) / 100.0) if r.get("Withdrawal_Rate") is not None else None
                        for r in rows
                    ],
                }

                # ---- Monte Carlo (TOP chart) -----------------------------------
                # Use SAME drift as deterministic (no +0.5*σ²) so it matches Live What-If.
                sigma = max(0.0, float(return_std))  # σ (decimal)

                # use same session seed as Live What-If so medians match
                seed = _get_or_create_seed()  # this is already defined below in the file

                mc_output = run_mc_with_seed(
                    seed,
                    run_monte_carlo_simulation_locked_inputs,
                    current_age=int(current_age),
                    retirement_age=int(retirement_age),
                    annual_saving=float(monthly_saving_ui) * 12.0,
                    saving_increase_rate=float(saving_increase_rate),
                    current_assets=float(current_assets),

                    # pass through deterministic CAGR (NO +0.5σ² bump)
                    return_mean=float(return_rate),
                    return_mean_after=float(return_rate_after),
                    return_std=sigma,

                    annual_expense=float(monthly_living_expense) * 12.0,
                    inflation_mean=float(inflation_rate),
                    inflation_std=float(inflation_std),
                    cpp_monthly=float(cpp_monthly),
                    cpp_start_age=int(cpp_from),
                    cpp_end_age=int(cpp_to),
                    asset_liquidations=asset_liquidation,
                    life_expectancy=life_expectancy,
                    income_tax_rate=float(income_tax_rate),
                    num_simulations=300,  # keep 300 here
                )

                monte_carlo_data = {
                    "Age": mc_output.get("ages", []),
                    "Percentile_10": (mc_output.get("percentiles", {}) or {}).get("p10", []),
                    "Percentile_50": (mc_output.get("percentiles", {}) or {}).get("p50", []),
                    "Percentile_90": (mc_output.get("percentiles", {}) or {}).get("p90", []),
                }
                dp = mc_output.get("depletion_probs", {}) or {}
                depletion_stats = {
                    "age_75": float(dp.get(75, 0.0)),
                    "age_85": float(dp.get(85, 0.0)),
                    "age_90": float(dp.get(90, 0.0)),
                    "ever": float(dp.get("ever", 0.0)),
                }

            except Exception as e:
                # Keep page rendering even if something went wrong
                current_app.logger.exception("retirement POST failed: %s", e)
                result = None
                table = []
                # chart_data, monte_carlo_data, depletion_stats remain defaults

    # Load saved scenarios for the user
    selected_scenario_id = request.form.get("load_scenario_select", "")
    saved_scenarios = (
        RetirementScenario.query.filter_by(user_id=current_user.id).all()
        if current_user.is_authenticated else []
    )

    # Render the template with all data
    return render_template(
        "retirement.html",
        result=result,
        table=table,
        table_headers=table_headers,
        retirement_age=retirement_age,
        reset=reset,
        chart_data=chart_data,
        monte_carlo_data=monte_carlo_data,
        depletion_stats=depletion_stats,
        return_std=request.form.get("return_std", "8"),
        inflation_std=request.form.get("inflation_std", "0.5"),
        selected_scenario_id=selected_scenario_id,
        saved_scenarios=saved_scenarios,
        sensitivities=sensitivities,
        sensitivity_headers=sensitivity_headers,
        sensitivity_table=sensitivity_table,
    )




# ===== New Scenario Blueprint and Routes =====
import logging
import re
from secrets import randbits

from flask import Blueprint, request, jsonify, current_app, session
from flask_login import login_required, current_user
from sqlalchemy import func

from models import db
from models.retirement.retirement_scenario import RetirementScenario

# If you have a separate projects_bp, you can register the compare route there.
# Otherwise we’ll attach it to scenarios_bp.
scenarios_bp = Blueprint("scenarios", __name__, url_prefix="/scenarios")
logger = logging.getLogger(__name__)
projects_bp = scenarios_bp  # safe default; replace with your real projects_bp if needed


# -------------------------
# Canonicalization helpers
# -------------------------
RATE_KEYS = {
    "return_rate", "return_rate_after",
    "inflation_rate", "saving_increase_rate",
    "income_tax_rate",
    "return_mean", "return_mean_after", "return_std", "inflation_std",
}

INT_LIKE = re.compile(r"(age|year|iter|seed|horizon|projection|expectancy)$", re.I)
RATE_LIKE = re.compile(r"(rate|mean|std)$", re.I)

DEFAULTS = {
    "current_age": 40,
    "retirement_age": 65,
    "life_expectancy": 84,        # <- critical to avoid KeyError
    "current_assets": 0.0,
    "annual_saving": 0.0,
    "annual_expense": 0.0,
    "return_rate": 0.06,
    "return_rate_after": 0.04,
    "inflation_rate": 0.025,
    "saving_increase_rate": 0.00,
    "income_tax_rate": 0.15,
    "cpp_monthly": 0.0,
    "cpp_start_age": None,        # will be defaulted to retirement_age
    "cpp_end_age": None,          # will be defaulted to life_expectancy
    "return_std": 0.10,
    "inflation_std": 0.005,
}

def _clean_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).replace(",", "").strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except Exception:
        return x

def _coerce_types(d: dict) -> dict:
    """Coerce strings to numbers; ints for age-like keys; floats elsewhere."""
    out = {}
    for k, v in (d or {}).items():
        v2 = _clean_number(v)
        if isinstance(v2, (int, float)):
            if INT_LIKE.search(k):
                out[k] = int(round(v2))
            else:
                out[k] = float(v2)
        else:
            out[k] = v2
    return out

def _normalize_rates(d: dict) -> dict:
    """Convert whole percents to decimals for keys that look like rates/means/stds."""
    out = dict(d)
    for k, v in list(out.items()):
        if isinstance(v, (int, float)) and (k in RATE_KEYS or RATE_LIKE.search(k)):
            # Treat obvious whole percentages as decimals
            if 2 <= v <= 1000:
                out[k] = v / 100.0
    return out

def to_canonical_inputs(raw: dict) -> dict:
    """
    Robust canonicalizer:
      - coerce numbers
      - normalize percent-like to decimals
      - apply sane defaults (DO NOT override existing)
      - fill missing CPP window and horizons
    """
    d = _coerce_types(raw or {})
    d = _normalize_rates(d)

    # Apply defaults for missing keys only
    for k, v in DEFAULTS.items():
        d.setdefault(k, v)

    # If user didn't specify CPP window, derive from horizon
    if d.get("cpp_start_age") is None:
        d["cpp_start_age"] = int(d["retirement_age"])
    if d.get("cpp_end_age") is None:
        d["cpp_end_age"] = int(d["life_expectancy"])

    # Horizon sanity
    if int(d["life_expectancy"]) < int(d["retirement_age"]):
        d["life_expectancy"] = int(d["retirement_age"])

    return d

def canonical_to_form_inputs(d: dict) -> dict:
    """
    Convert canonical (annual) values back to your form fields:
      - expose monthly_living_expense (annual_expense / 12)
      - show monthly contribution in 'annual_saving' field (divide by 12)
    """
    if not isinstance(d, dict):
        return {}
    c = to_canonical_inputs(d)
    out = dict(c)
    out["monthly_living_expense"] = float(c.get("annual_expense", 0.0)) / 12.0
    out["annual_saving"] = float(c.get("annual_saving", 0.0)) / 12.0
    out["_units"] = "form"
    return out

def _get_or_create_seed() -> int:
    try:
        seed = session.get("ret_mc_seed")
        if seed is None:
            if getattr(current_user, "is_authenticated", False):
                seed = abs(hash(f"mcseed::{current_user.get_id()}")) % (2**31 - 1)
            else:
                seed = randbits(31)
            session["ret_mc_seed"] = int(seed)
        return int(seed)
    except Exception:
        return int(randbits(31))


# =======================
#   CRUD for scenarios
# =======================
@scenarios_bp.route("/save", methods=["POST"])
@login_required
def save_scenario():
    data = request.get_json(silent=True) or {}
    scenario_name = data.get("scenario_name")
    inputs_json   = (data.get("inputs_json") or {}).copy()

    if not scenario_name or inputs_json is None:
        return jsonify({"error": "Missing scenario_name or inputs_json"}), 400

    # Only convert if the payload is the MONTHLY form variant
    units_hint    = (data.get("units") or inputs_json.get("_units") or "").lower()
    is_form_units = (
        units_hint == "form"
        or ("monthly_living_expense" in inputs_json and "annual_expense" not in inputs_json)
    )
    if is_form_units:
        # monthly -> annual
        try:
            inputs_json["annual_saving"] = _clean_number(inputs_json.get("annual_saving", 0.0)) * 12.0
        except Exception:
            inputs_json["annual_saving"] = 0.0
        try:
            ml = _clean_number(inputs_json.pop("monthly_living_expense", 0.0))
            inputs_json["annual_expense"] = ml * 12.0
        except Exception:
            inputs_json.pop("monthly_living_expense", None)
            inputs_json["annual_expense"] = _clean_number(inputs_json.get("annual_expense", 0.0))
        inputs_json["_units"] = "canonical"

    canon = to_canonical_inputs(inputs_json)

    existing = RetirementScenario.query.filter_by(
        user_id=current_user.id, scenario_name=scenario_name
    ).first()

    if existing:
        existing.inputs_json = canon
    else:
        db.session.add(
            RetirementScenario(
                user_id=current_user.id,
                scenario_name=scenario_name,
                inputs_json=canon
            )
        )
    db.session.commit()
    return jsonify({"message": "Scenario saved successfully."}), 200


@scenarios_bp.route("/list", methods=["GET"])
@login_required
def list_scenarios():
    scenarios = RetirementScenario.query.filter_by(user_id=current_user.id).all()
    result = [
        {
            "id": s.id,
            "scenario_name": s.scenario_name,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in scenarios
    ]
    return jsonify(result), 200


@scenarios_bp.route("/load/<int:scenario_id>", methods=["GET"])
@login_required
def load_scenario(scenario_id):
    scenario = RetirementScenario.query.filter_by(id=scenario_id, user_id=current_user.id).first()
    if not scenario:
        return jsonify({"error": "Scenario not found"}), 404

    # Convert canonical -> form units for the UI
    form_inputs = canonical_to_form_inputs(scenario.inputs_json or {})

    return jsonify(
        {
            "scenario_name": scenario.scenario_name,
            "inputs_json": form_inputs,
            "created_at": scenario.created_at.isoformat(),
            "updated_at": scenario.updated_at.isoformat(),
        }
    ), 200


@scenarios_bp.route("/delete/<int:scenario_id>", methods=["DELETE"])
@login_required
def delete_scenario(scenario_id):
    scenario = RetirementScenario.query.filter_by(id=scenario_id, user_id=current_user.id).first()
    if not scenario:
        return jsonify({"error": "Scenario not found"}), 404

    try:
        db.session.delete(scenario)
        db.session.commit()
        return jsonify({"message": f"Scenario '{scenario.scenario_name}' deleted successfully."}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to delete scenario.", "details": str(e)}), 500


# -------------------------------
# Compare Two Scenarios (A vs. B)
# -------------------------------
_PROJECTION_KEYS = {
    "current_age","retirement_age","annual_saving","saving_increase_rate",
    "current_assets","return_rate","return_rate_after","annual_expense",
    "cpp_monthly","cpp_start_age","cpp_end_age","asset_liquidations",
    "inflation_rate","life_expectancy","income_tax_rate"
}

def _projection_args_from_params(p):
    return {k: p[k] for k in _PROJECTION_KEYS if k in p}

def _mc_args_from_params(p):
    return {
        "current_age":          p["current_age"],
        "retirement_age":       p["retirement_age"],
        "annual_saving":        p["annual_saving"],
        "saving_increase_rate": p["saving_increase_rate"],
        "current_assets":       p["current_assets"],
        "return_mean":          p["return_rate"],       # pass CAGR through
        "return_mean_after":    p["return_rate_after"],
        "return_std":           p.get("return_std", DEFAULTS["return_std"]),
        "annual_expense":       p["annual_expense"],
        "inflation_mean":       p["inflation_rate"],
        "inflation_std":        p.get("inflation_std", DEFAULTS["inflation_std"]),
        "cpp_monthly":          p.get("cpp_monthly", 0.0),
        "cpp_start_age":        p.get("cpp_start_age", p["retirement_age"]),
        "cpp_end_age":          p.get("cpp_end_age", p["life_expectancy"]),
        "asset_liquidations":   p.get("asset_liquidations", []),
        "life_expectancy":      p["life_expectancy"],
        "income_tax_rate":      p.get("income_tax_rate", DEFAULTS["income_tax_rate"]),
        "num_simulations":      300,
    }

def _resolve_scenario(ref):
    """Allow selecting by numeric id or by scenario_name (case-insensitive)."""
    if ref is None:
        return None
    sref = str(ref).strip()
    if not sref:
        return None
    # try numeric id
    try:
        sid = int(sref)
        s = RetirementScenario.query.get(sid)
        if s and current_user.is_authenticated and s.user_id != current_user.id:
            return None
        if s:
            return s
    except Exception:
        pass
    # fall back to name
    q = RetirementScenario.query
    if current_user.is_authenticated:
        q = q.filter(RetirementScenario.user_id == current_user.id)
    return (
        q.filter(func.lower(RetirementScenario.scenario_name) == sref.lower())
         .order_by(RetirementScenario.updated_at.desc())
         .first()
    )

@projects_bp.route("/retirement/compare", methods=["POST"])
def compare_retirement():
    def jerr(msg, code=400, extra=None):
        if extra:
            try:
                current_app.logger.warning("compare_retirement: %s | extra=%s", msg, extra)
            except Exception:
                pass
        out = {"error": msg}
        if current_app and current_app.debug and extra:
            out["_extra"] = extra
        return jsonify(out), code

    try:
        data_json = request.get_json(silent=True) or {}
        raw_a = (request.form.get("scenario_a") or data_json.get("scenario_a") or "").strip()
        raw_b = (request.form.get("scenario_b") or data_json.get("scenario_b") or "").strip()

        scen_a = _resolve_scenario(raw_a)
        if not scen_a:
            return jerr("Scenario A not found (id or name).", 404, {"scenario_a": raw_a})
        scen_b = _resolve_scenario(raw_b) if raw_b else None
        if raw_b and not scen_b:
            return jerr("Scenario B not found (id or name).", 404, {"scenario_b": raw_b})

        if current_user.is_authenticated:
            if scen_a.user_id != current_user.id or (scen_b and scen_b.user_id != current_user.id):
                return jerr("You can only compare your own scenarios.", 403)

        # Helpers
        def _to_map(ages, values):
            return {int(a): (float(v) if v is not None else None) for a, v in zip(ages, values)}

        seed = _get_or_create_seed()

        from models.retirement.retirement_calc import (
            run_monte_carlo_simulation_locked_inputs,
            run_mc_with_seed,
            sensitivity_analysis,
        )

        def _run_mc_for(scn):
            raw = scn.inputs_json or {}
            p = to_canonical_inputs(raw)     # <- robust normalization + defaults (life_expectancy!)
            mc_args = _mc_args_from_params(p)

            if current_app and current_app.debug:
                current_app.logger.info("COMPARE mc_args for '%s' [id=%s]: %s", scn.scenario_name, scn.id, mc_args)

            mc = run_mc_with_seed(seed, run_monte_carlo_simulation_locked_inputs, **mc_args)

            ages = [int(x) for x in mc.get("ages", [])]
            pcts = mc.get("percentiles", {}) or {}
            p10  = [float(x) for x in pcts.get("p10", [])]
            p50  = [float(x) for x in pcts.get("p50", [])]
            p90  = [float(x) for x in pcts.get("p90", [])]
            n = min(len(ages), len(p10), len(p50), len(p90))

            start_age = mc_args.get("current_age") or (ages[0] if ages else None)
            end_age   = mc_args.get("life_expectancy") or (ages[-1] if ages else None)

            return {
                "label": scn.scenario_name,
                "ages": ages[:n],
                "p10":  p10[:n],
                "p50":  p50[:n],
                "p90":  p90[:n],
                "_args": mc_args,
                "start_age": int(start_age),
                "end_age": int(end_age),
                "_scenario_id": scn.id,
                "_canon": p,
                "_raw": raw,
            }

        # Run A/B
        try:
            A = _run_mc_for(scen_a)
        except Exception as e:
            try:
                current_app.logger.exception("compare_retirement A failed")
            except Exception:
                logger.exception("compare_retirement A failed")
            return jerr(f"MC failed for scenario '{scen_a.scenario_name}': {e}", 400)

        B = None
        if scen_b:
            try:
                B = _run_mc_for(scen_b)
            except Exception as e:
                try:
                    current_app.logger.exception("compare_retirement B failed")
                except Exception:
                    logger.exception("compare_retirement B failed")
                return jerr(f"MC failed for scenario '{scen_b.scenario_name}': {e}", 400)

        # Build union axis
        axis_start = A["start_age"]
        axis_end   = A["end_age"]
        if B:
            axis_start = min(axis_start, B["start_age"])
            axis_end   = max(axis_end,   B["end_age"])
        axis_ages = list(range(int(axis_start), int(axis_end) + 1))

        Amap10, Amap50, Amap90 = _to_map(A["ages"], A["p10"]), _to_map(A["ages"], A["p50"]), _to_map(A["ages"], A["p90"])
        A_p10 = [Amap10.get(a) for a in axis_ages]
        A_p50 = [Amap50.get(a) for a in axis_ages]
        A_p90 = [Amap90.get(a) for a in axis_ages]

        payload = {
            "labels": {"A": A["label"]},
            "mc": {
                "ages": axis_ages,
                "p10": {"A": A_p10},
                "p50": {"A": A_p50},
                "p90": {"A": A_p90},
            },
            "meta": {"A": {"label": A["label"], "start_age": A["start_age"], "end_age": A["end_age"]}},
        }
        warning = None

        if B:
            Bmap10, Bmap50, Bmap90 = _to_map(B["ages"], B["p10"]), _to_map(B["ages"], B["p50"]), _to_map(B["ages"], B["p90"])
            payload["mc"]["p10"]["B"] = [Bmap10.get(a) for a in axis_ages]
            payload["mc"]["p50"]["B"] = [Bmap50.get(a) for a in axis_ages]
            payload["mc"]["p90"]["B"] = [Bmap90.get(a) for a in axis_ages]
            payload["labels"]["B"] = B["label"]
            payload["meta"]["B"] = {"label": B["label"], "start_age": B["start_age"], "end_age": B["end_age"]}

            def _max_or_zero(arr):
                vals = [x for x in arr if x is not None]
                return max(vals) if vals else 0
            if _max_or_zero(payload["mc"]["p50"]["B"]) > 1e9 and _max_or_zero(A_p50) > 0:
                warning = "Scenario B may be using different units (rates/years). Overlay shown; please review inputs."

        # Sensitivity (deterministic)
        SENS_VARS = [
            "current_assets", "return_rate", "return_rate_after",
            "annual_saving", "annual_expense", "saving_increase_rate",
            "inflation_rate", "income_tax_rate", "retirement_age",
        ]
        def _proj_args_for(scn):
            params = to_canonical_inputs(scn.inputs_json or {})
            return _projection_args_from_params(params)

        sensA_dollar, sensA_pct = [], []
        sensB_dollar, sensB_pct = None, None
        try:
            from models.retirement.retirement_calc import sensitivity_analysis
            sA = sensitivity_analysis(_proj_args_for(scen_a), SENS_VARS, delta=0.01)
            sensA_dollar = [sA[v]["dollar_impact"] if v in sA else 0 for v in SENS_VARS]
            sensA_pct    = [sA[v]["sensitivity_pct"] if v in sA else 0 for v in SENS_VARS]
        except Exception as e:
            try:
                current_app.logger.exception("Sensitivity A failed: %s", e)
            except Exception:
                logger.exception("Sensitivity A failed: %s", e)

        if scen_b:
            try:
                from models.retirement.retirement_calc import sensitivity_analysis
                sB = sensitivity_analysis(_proj_args_for(scen_b), SENS_VARS, delta=0.01)
                sensB_dollar = [sB[v]["dollar_impact"] if v in sB else 0 for v in SENS_VARS]
                sensB_pct    = [sB[v]["sensitivity_pct"] if v in sB else 0 for v in SENS_VARS]
            except Exception as e:
                try:
                    current_app.logger.exception("Sensitivity B failed: %s", e)
                except Exception:
                    logger.exception("Sensitivity B failed: %s", e)

        payload["sens"] = {"vars": SENS_VARS, "A": sensA_dollar, "A_pct": sensA_pct}
        if sensB_dollar is not None:
            payload["sens"]["B"] = sensB_dollar
            payload["sens"]["B_pct"] = sensB_pct
        if warning:
            payload["warning"] = warning

        if (current_app and current_app.debug) or (current_app and current_app.config.get("COMPARE_DEBUG")):
            payload["_debug"] = {
                "seed": _get_or_create_seed(),
                "A_id": A.get("_scenario_id"),
                "A_label": A["label"],
                "A_raw": A.get("_raw"),
                "A_canon": A.get("_canon"),
                "A_args": A.get("_args"),
                **({
                    "B_id": B.get("_scenario_id"),
                    "B_label": B["label"],
                    "B_raw": B.get("_raw"),
                    "B_canon": B.get("_canon"),
                    "B_args": B.get("_args"),
                } if B else {})
            }

        return jsonify(payload), 200

    except Exception as e:
        try:
            current_app.logger.exception("compare_retirement unexpected failure: %s", e)
        except Exception:
            logger.exception("compare_retirement unexpected failure: %s", e)
        msg = "Server error during compare."
        if current_app and current_app.debug:
            msg += f" ({e})"
        return jsonify({"error": msg}), 500











# ==== Live-WhatIf: minimal POST endpoint (append-only) ====
from flask import request, session, jsonify
import secrets

# Reuse your projects blueprint if it exists; otherwise create a tiny one.
try:
    bp_for_live = projects_bp
except NameError:
    from flask import Blueprint
    bp_for_live = Blueprint("live_whatif", __name__)

# Seed wrapper
from models.retirement.retirement_calc import run_mc_with_seed

# --- 🔁 USE YOUR REAL FUNCTION NAMES & MODULE PATH ---
from models.retirement.retirement_calc import run_retirement_projection as _DET
from models.retirement.retirement_calc import run_monte_carlo_simulation_locked_inputs as _MC
# ----------------------------------------------------

def _get_or_create_seed():
    if "mc_seed" not in session:
        session["mc_seed"] = secrets.randbits(32)
    return int(session["mc_seed"])

def _defaults():
    # Must include required params for BOTH deterministic + MC
    return dict(
        # shared
        current_age=53,
        retirement_age=65,
        annual_saving=48000,
        saving_increase_rate=0.02,
        current_assets=850000,
        annual_expense=72000,
        cpp_monthly=1200,
        cpp_start_age=65,
        cpp_end_age=70,
        asset_liquidations=[],           # e.g. [{"age": 60, "amount": 100000}]
        inflation_rate=0.025,
        life_expectancy=92,
        income_tax_rate=0.15,

        # deterministic-specific
        return_rate=0.065,
        return_rate_after=0.045,

        # MC-specific (mapped below)
        return_mean=0.065,
        return_mean_after=0.045,
        return_std=0.10,
        inflation_mean=0.025,
        inflation_std=0.005,
        num_simulations=300,   # ← unified to 300
    )

def _merge(d):
    base = _defaults()
    for k, v in (d or {}).items():
        if k in base:
            base[k] = v
    return base

def _harmonize(params: dict) -> dict:
    """
    If the UI doesn't provide an explicit CPP window, infer a window that
    matches the main calculator: start at retirement, end at life expectancy.
    """
    ra = int(params.get("retirement_age", 65))
    le = int(params.get("life_expectancy", ra + 30))
    if "cpp_start_age" not in params or params.get("cpp_start_age") is None:
        params["cpp_start_age"] = ra
    if "cpp_end_age" not in params or params.get("cpp_end_age") is None:
        params["cpp_end_age"] = le
    return params



# Goals helpers
from models.retirement.goals import expand_goals_to_per_age, goals_to_liquidations_adapter
# Coach helpers (still available, optional)
from models.retirement.coach import coach_suggestions

# ---------- shared helpers (coalesce + goal merge) ----------
def _coalesce_liqs(liqs: list[dict]) -> list[dict]:
    """Coalesce liquidations by age exactly like live_update already does."""
    if not liqs:
        return []
    liqs = sorted(liqs, key=lambda r: int(r.get("age", 0)))
    out: list[dict] = []
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

def _merge_goals_into_liqs(params: dict) -> list[dict]:
    """
    Expand params['goal_events'] to per-age and merge into existing
    params['asset_liquidations'], returning a coalesced list.
    Safe to call even if there are no goals.
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
        # Never fail hard on goal merge during solver/live calc
        return base_liqs


# --- tiny helpers for the solver & sims ---
def _build_det_args(p):
    keys = [
        "current_age", "retirement_age", "annual_saving", "saving_increase_rate",
        "current_assets", "return_rate", "return_rate_after", "annual_expense",
        "cpp_monthly", "cpp_start_age", "cpp_end_age",
        "asset_liquidations", "inflation_rate", "life_expectancy", "income_tax_rate",
    ]
    return {k: p[k] for k in keys if k in p}

def _arith_from_cagr(cagr: float, sigma: float) -> float:
    # for lognormal step: mu = CAGR + 0.5 * sigma^2
    return float(cagr) + 0.5 * float(sigma) * float(sigma)

def _build_mc_args(p, n_sims):
    std = float(p["return_std"])
    # Pass through deterministic CAGR directly (NO +0.5σ² bump)
    mean_pre  = float(p["return_rate"])
    mean_post = float(p["return_rate_after"])
    return dict(
        current_age=int(p["current_age"]),
        retirement_age=int(p["retirement_age"]),
        annual_saving=float(p["annual_saving"]),
        saving_increase_rate=float(p["saving_increase_rate"]),
        current_assets=float(p["current_assets"]),
        return_mean=mean_pre,
        return_mean_after=mean_post,
        return_std=std,
        annual_expense=float(p["annual_expense"]),
        inflation_mean=float(p.get("inflation_mean", p["inflation_rate"])),
        inflation_std=float(p["inflation_std"]),
        cpp_monthly=float(p["cpp_monthly"]),
        cpp_start_age=int(p["cpp_start_age"]),
        cpp_end_age=int(p["cpp_end_age"]),
        asset_liquidations=list(p.get("asset_liquidations") or []),
        life_expectancy=int(p["life_expectancy"]),
        num_simulations=int(n_sims),
        income_tax_rate=float(p["income_tax_rate"]),
    )

def _series_value_for(metric, det_curve, pct, idx):
    """Safe value lookup: clamp idx to the last known point to avoid None when target age > horizon."""
    if idx < 0:
        return None
    if metric == "det":
        if not det_curve:
            return None
        i = min(idx, len(det_curve) - 1)
        return det_curve[i]
    key = {"p10":"p10", "p50":"p50", "p90":"p90"}[metric]
    seq = pct.get(key) or []
    if not seq:
        return None
    i = min(idx, len(seq) - 1)
    return seq[i]

def _solve_single_lever(p0, prefs, seed, _DET, _MC, run_mc_with_seed):
    """
    prefs:
      { "target": { "metric": "det|p10|p50|p90", "assets": 2000000, "age": 90 },
        "lever":  "retirement_age|annual_expense|annual_saving|post_ret_return",
      }
    Returns {"patch": {...}, "achieved": float, "lever_value": value, "summary": str} or {}
    """
    target = (prefs or {}).get("target") or {}
    metric = target.get("metric") or "p50"
    want_assets = float(target.get("assets") or 0.0)

    # Clamp target age to horizon so asking for age 90 with life=84 doesn't read as None/0
    start_age = int(p0["current_age"])
    life = int(p0["life_expectancy"])
    req_age = int(target.get("age") or life)
    target_age = max(start_age, min(req_age, life))

    lever = (prefs or {}).get("lever") or "retirement_age"

    # Build a quick evaluator (mirrors server live path + goal merge)
    def eval_assets(p):
        # Ensure any goal_events are reflected as liquidations for the sim
        p_eval = dict(p)
        p_eval["asset_liquidations"] = _merge_goals_into_liqs(p_eval)

        # deterministic
        det_out = _DET(**_build_det_args(p_eval))
        det_tbl = det_out.get("table", [])
        det_curve = [row.get("Asset") for row in det_tbl]

        # fast MC — honor requested num_simulations if present (default 300)
        n_sims = int(p_eval.get("num_simulations", 300))
        mc = run_mc_with_seed(seed, _MC, **_build_mc_args(p_eval, n_sims=n_sims))
        pct = mc.get("percentiles", {})

        idx = target_age - start_age
        return _series_value_for(metric, det_curve, pct, idx)

    base_val = eval_assets(dict(p0))
    # If we already exceed target, return no-op patch
    if base_val is not None and base_val >= want_assets:
        return {
            "patch": {},
            "achieved": float(base_val),
            "lever_value": p0.get(lever),
            "summary": f"Already meets target ({metric} @ {target_age} = ${base_val:,.0f})"
        }

    # Define ranges & direction per lever (monotonic assumptions)
    low, high, inc = None, None, +1

    def apply_lever(p, x):
        if lever == "retirement_age":
            p["retirement_age"] = int(round(x))
        elif lever == "annual_expense":
            p["annual_expense"] = max(0.0, float(x))
        elif lever == "annual_saving":
            p["annual_saving"] = max(0.0, float(x))
        elif lever == "post_ret_return":
            xr = max(0.0, float(x))  # decimal (e.g., 0.06 for 6%)
            p["return_rate_after"] = xr
            p["return_mean_after"] = xr
        return p

    if lever == "retirement_age":
        # Honor horizon and current age; never allow retiring at/after life expectancy
        low  = max(int(p0["current_age"]), 50)
        high = min(80, max(int(p0["current_age"]) + 1, life - 1))
        inc  = +1  # increasing age increases assets
    elif lever == "annual_expense":
        low, high, inc = 0.0, float(p0["annual_expense"]) * 1.5, -1  # more spend ↓ assets
    elif lever == "annual_saving":
        low, high, inc = 0.0, max(float(p0["annual_saving"]) * 2.0, 120000.0), +1
    elif lever == "post_ret_return":
        low, high, inc = 0.0, 0.12, +1  # 0% .. 12% post-ret return (decimal)
    else:
        return {}

    # Check attainability
    p_low  = apply_lever(dict(p0), low)
    v_low  = eval_assets(p_low)
    p_high = apply_lever(dict(p0), high)
    v_high = eval_assets(p_high)

    # Normalize monotonic orientation so "low" is below target and "high" can reach it
    def below(v): return (v is None) or (v < want_assets)
    def above(v): return (v is not None) and (v >= want_assets)

    feasible = True
    if inc > 0:
        if above(v_low):   # already enough at low bound
            high = low
        elif not above(v_high):
            feasible = False
    else:
        # decreasing lever value improves assets (spending)
        if above(v_high):  # already enough at higher spend (unlikely)
            low = high
        elif not above(v_low):
            feasible = False

    if not feasible:
        # pick closer bound
        pick = (high if (abs((v_high or 0) - want_assets) <= abs((v_low or 0) - want_assets)) else low)
        ach = eval_assets(apply_lever(dict(p0), pick))
        return {"patch": apply_lever({}, pick), "achieved": float(ach or 0.0),
                "lever_value": pick, "summary": "Target not reachable within bounds; using nearest bound."}

    # Binary search
    best_x, best_v = None, None
    max_iter = 16
    tolX = (
        0.25 if lever == "retirement_age"
        else 0.0005 if lever == "post_ret_return"  # ~0.05%
        else 50.0                                   # ~$50 annual
    )
    tolF = 500.0  # $500 tolerance on assets

    lo, hi = low, high
    while max_iter > 0 and abs(hi - lo) > tolX:
        max_iter -= 1
        mid = (lo + hi) / 2.0
        v_mid = eval_assets(apply_lever(dict(p0), mid))
        if (v_mid is not None) and (v_mid >= want_assets):
            best_x, best_v = mid, v_mid
            if inc > 0: hi = mid
            else:       lo = mid
        else:
            if inc > 0: lo = mid
            else:       hi = mid
        if v_mid is not None and abs(v_mid - want_assets) < tolF:
            lo = hi = mid
            break

    xStar = (best_x if best_x is not None else (hi if inc > 0 else lo))
    vStar = (best_v if best_v is not None else eval_assets(apply_lever(dict(p0), xStar)))

    # Round/format lever value
    patch = {}
    if lever == "retirement_age":
        patch["retirement_age"] = int(round(xStar))
    elif lever == "annual_expense":
        patch["annual_expense"] = round(float(xStar), 2)
    elif lever == "annual_saving":
        patch["annual_saving"] = round(float(xStar), 2)
    elif lever == "post_ret_return":
        xr = float(xStar)
        patch["return_rate_after"] = xr
        patch["return_mean_after"] = xr

    return {
        "patch": patch,
        "achieved": float(vStar or 0.0),
        "lever_value": (
            patch.get("retirement_age") if lever == "retirement_age"
            else patch.get("annual_expense") if lever == "annual_expense"
            else patch.get("annual_saving") if lever == "annual_saving"
            else patch.get("return_rate_after") if lever == "post_ret_return"
            else None
        ),
        "summary": f"{lever.replace('_',' ')} → {patch} gives {metric} @ {target_age} ≈ ${float(vStar or 0.0):,.0f}"
    }


@bp_for_live.post("/api/live-update")
def live_update():
    payload = request.get_json(silent=True) or {}

    # speed mode: "lite" while dragging, "full" after user pauses or presses Solve
    mode = payload.get("mode", "full")
    is_lite = (mode == "lite")

    # optional target/lever preferences for the solver/coach
    coach_prefs = payload.get("coach_prefs", {}) or {}

    params = _merge(payload)
    params = _harmonize(params)  # keep CPP window aligned with main calculator

    goal_events = params.get("goal_events") or []

    # ---- Expand goals to per-age cashflows (post-tax via liquidations) ----
    per_age = {}
    goals_error = None
    try:
        already_merged = bool(params.get("asset_liquidations"))
        if goal_events and not already_merged:
            # Use the same merge+coalesce the coach uses
            params["asset_liquidations"] = _merge_goals_into_liqs(params)
    except Exception as e:
        goals_error = str(e)
        params["asset_liquidations"] = list(params.get("asset_liquidations") or [])

    # ---- Deterministic ----
    det_out = _DET(**_build_det_args(params))
    det_table = det_out.get("table", [])
    labels = [row.get("Year") for row in det_table]
    det_curve = [row.get("Asset") for row in det_table]

    # ---- Monte Carlo ----
    mc_params = _build_mc_args(
        params,
        n_sims=300  # ← unified to 300 for both lite and full
    )
    seed = _get_or_create_seed()
    mc_out = run_mc_with_seed(seed, _MC, **mc_params)
    pct = mc_out.get("percentiles", {})
    det_last = det_curve[-1] if det_curve else None

    # ---- Targeted solve (single lever) when FULL + prefs present ----
    solution = {}
    if not is_lite and coach_prefs:
        try:
            # expected schema: {"target":{"metric":"det|p10|p50|p90","assets":..., "age":...}, "lever":...}
            solution = _solve_single_lever(params, coach_prefs, seed, _DET, _MC, run_mc_with_seed)
        except Exception as e:
            solution = {"error": str(e)}

    # ---- Coach suggestions (optional) only on FULL ----
    coach = []
    if not is_lite:
        try:
            coach = coach_suggestions(params, pct, prefs=coach_prefs)
        except Exception as e:
            coach = [{"type": "status", "title": "Coach unavailable", "detail": str(e)}]

    debug = {
        "used_params": {
            "current_age": params["current_age"],
            "retirement_age": params["retirement_age"],
            "life_expectancy": params["life_expectancy"],
            "annual_saving": params["annual_saving"],
            "saving_increase_rate": params["saving_increase_rate"],
            "current_assets": params["current_assets"],
            "annual_expense": params["annual_expense"],
            "income_tax_rate": params["income_tax_rate"],
            "inflation_rate": params["inflation_rate"],
            "cpp_monthly": params["cpp_monthly"],
            "cpp_start_age": params["cpp_start_age"],
            "cpp_end_age": params["cpp_end_age"],
            "return_rate": params["return_rate"],
            "return_rate_after": params["return_rate_after"],
            "return_std": params["return_std"],
            "inflation_std": params["inflation_std"],
            "asset_liquidations": params["asset_liquidations"],
            "goal_events_count": len(goal_events),
            "mode": mode,
            "num_simulations": mc_params["num_simulations"],
        },
        "goals": {"per_age": per_age, "error": goals_error},
        "mc": {"seed": seed},
        "derived": {"deterministic_last": det_last},
        "coach_prefs": coach_prefs if not is_lite else {},
        "solution": solution,
    }

    out = {
        "labels": labels,
        "deterministic": det_curve,
        "p10": pct.get("p10", []),
        "p50": pct.get("p50", []),
        "p90": pct.get("p90", []),
        **({"coach": coach} if not is_lite else {}),
        **({"solution": solution} if not is_lite else {}),
        "debug": debug,
    }
    return jsonify(out), 200






















