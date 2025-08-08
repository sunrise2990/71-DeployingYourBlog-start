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
        "inflation_rate", "income_tax_rate", "cpp_monthly"
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
    depletion_stats = {
        "age_75": 0.0,
        "age_85": 0.0,
        "age_90": 0.0,
        "ever":   0.0
    }
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

    # Prepare sensitivity‐analysis headers and container
    sensitivity_headers = ["Variable", "Sensitivity (%)", "Dollar Impact ($)"]
    sensitivity_table = []

    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset":
            reset = True
        elif action == "calculate":
            try:
                # Helper to fetch+cast form inputs
                def get_form_value(name, cast_func, default=0):
                    val = request.form.get(name)
                    form_inputs[name] = val
                    return cast_func(val) if val else default

                # --- Gather form inputs ---
                current_age            = get_form_value("current_age", int)
                retirement_age         = get_form_value("retirement_age", int)
                monthly_saving         = get_form_value("annual_saving", float)  # form field name kept as-is
                return_rate            = get_form_value("return_rate", float) / 100
                return_rate_after      = get_form_value("return_rate_after", float) / 100
                lifespan               = get_form_value("lifespan", int)
                monthly_living_expense = get_form_value("monthly_living_expense", float)
                inflation_rate         = get_form_value("inflation_rate", float) / 100
                current_assets         = get_form_value("current_assets", float)
                saving_increase_rate   = get_form_value("saving_increase_rate", float) / 100
                income_tax_rate        = get_form_value("income_tax_rate", float) / 100

                cpp_monthly = get_form_value("cpp_support", float)
                cpp_from    = get_form_value("cpp_from_age", int)
                cpp_to      = get_form_value("cpp_to_age", int)

                return_std    = get_form_value("return_std", float) / 100
                inflation_std = get_form_value("inflation_std", float) / 100

                # --- Collect asset liquidation events ---
                asset_liquidation = []
                for i in range(1, 4):
                    amt = get_form_value(f"asset_liquidation_{i}", float)
                    age = get_form_value(f"asset_liquidation_age_{i}", int)
                    if amt != 0 and age > 0:
                        asset_liquidation.append({"amount": amt, "age": age})

                # --- Build baseline parameter dict ---
                baseline_params = {
                    "current_age": current_age,
                    "retirement_age": retirement_age,
                    "annual_saving": monthly_saving * 12,  # convert form value to annual
                    "saving_increase_rate": saving_increase_rate,
                    "current_assets": current_assets,
                    "return_rate": return_rate,
                    "return_rate_after": return_rate_after,
                    "annual_expense": monthly_living_expense * 12,
                    "cpp_monthly": cpp_monthly,
                    "cpp_start_age": cpp_from,
                    "cpp_end_age": cpp_to,
                    "asset_liquidations": asset_liquidation,
                    "inflation_rate": inflation_rate,
                    "life_expectancy": lifespan,
                    "income_tax_rate": income_tax_rate
                }

                # Variables to test in sensitivity analysis
                variables = [
                    "current_assets", "return_rate", "return_rate_after",
                    "annual_saving", "annual_expense", "saving_increase_rate",
                    "inflation_rate", "income_tax_rate"
                ]

                # Run sensitivity analysis (includes dollar impact)
                sensitivities = sensitivity_analysis(baseline_params, variables, delta=0.01)
                sensitivity_table = [
                    [
                        var,
                        f"{vals['sensitivity_pct']:.2f}%",
                        f"${vals['dollar_impact']:,.0f}"
                    ]
                    for var, vals in sensitivities.items()
                ]

                # --- Main projection run ---
                output = run_retirement_projection(**baseline_params)
                result = output["final_assets"]

                # Ensure Living_Exp_Retirement exists for every row
                for row in output["table"]:
                    if not row.get("Living_Exp_Retirement"):
                        row["Living_Exp_Retirement"] = row.get("Living_Exp", 0)

                # Build the projection table
                table = [[
                    row.get("Age"),
                    row.get("Year"),
                    row.get("Retire"),
                    f"${row.get('Living_Exp', 0):,.0f}",
                    f"${row.get('CPP_Support', 0):,.0f}" if row.get("CPP_Support") else "",
                    f"${row.get('Income_Tax_Payment', 0):,.0f}",
                    f"${row.get('Living_Exp_Retirement', 0):,.0f}",
                    f"${row.get('Asset_Liquidation', 0):,.0f}" if row.get("Asset_Liquidation") else "",
                    f"${row.get('Savings', 0):,.0f}" if row.get("Savings") else "",
                    f"${row.get('Asset', 0):,.0f}",
                    f"${row.get('Asset_Retirement', 0):,.0f}" if row.get("Asset_Retirement") else "",
                    f"${row.get('Investment_Return', 0):,.0f}" if row.get("Investment_Return") is not None else "",
                    f"{row.get('Return_Rate'):.1f}%" if row.get("Return_Rate") is not None else "",
                    f"{row.get('Withdrawal_Rate'):.1f}%" if row.get("Withdrawal_Rate") is not None else ""
                ] for row in output["table"]]

                # Prepare chart data
                chart_data = {
                    "Age": [r["Age"] for r in output["table"]],
                    "Living_Exp_Retirement": [r["Living_Exp_Retirement"] for r in output["table"]],
                    "Asset_Retirement": [r["Asset_Retirement"] or 0 for r in output["table"]],
                    "Withdrawal_Rate": [
                        (r["Withdrawal_Rate"] / 100) if r["Withdrawal_Rate"] else None
                        for r in output["table"]
                    ]
                }



                # Run Monte Carlo simulation
                mc_output = run_monte_carlo_simulation_locked_inputs(
                    current_age=current_age,
                    retirement_age=retirement_age,
                    annual_saving=monthly_saving * 12,
                    saving_increase_rate=saving_increase_rate,
                    current_assets=current_assets,
                    return_mean=return_rate,
                    return_mean_after=return_rate_after,
                    return_std=return_std,
                    annual_expense=monthly_living_expense * 12,
                    inflation_mean=inflation_rate,
                    inflation_std=inflation_std,
                    cpp_monthly=cpp_monthly,
                    cpp_start_age=cpp_from,
                    cpp_end_age=cpp_to,
                    asset_liquidations=asset_liquidation,
                    life_expectancy=lifespan,
                    income_tax_rate=income_tax_rate,
                    num_simulations=1000
                )
                monte_carlo_data = {
                    "Age": mc_output["ages"],
                    "Percentile_10": mc_output["percentiles"]["p10"],
                    "Percentile_50": mc_output["percentiles"]["p50"],
                    "Percentile_90": mc_output["percentiles"]["p90"]
                }
                depletion_stats = {
                    "age_75": mc_output["depletion_probs"].get(75, 0.0),
                    "age_85": mc_output["depletion_probs"].get(85, 0.0),
                    "age_90": mc_output["depletion_probs"].get(90, 0.0),
                    "ever":   mc_output["depletion_probs"].get("ever", 0.0)
                }

            except Exception as e:
                print("❌ Error in retirement projection:", e)
                result = None
                table = []
                # chart_data, monte_carlo_data, depletion_stats remain at default

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
        sensitivity_table=sensitivity_table
    )



# ===== New Scenario Blueprint and Routes =====

import logging
import traceback

scenarios_bp = Blueprint("scenarios", __name__, url_prefix="/scenarios")
logger = logging.getLogger(__name__)

@scenarios_bp.route("/save", methods=["POST"])
@login_required
def save_scenario():
    data = request.get_json(silent=True) or {}
    scenario_name = data.get("scenario_name")
    inputs_json   = data.get("inputs_json")  # may be legacy or canonical

    if not scenario_name or inputs_json is None:
        return jsonify({"error": "Missing scenario_name or inputs_json"}), 400

    # >>> ADD THIS: treat form's "Monthly Savings" as monthly, convert to annual for storage
    try:
        if isinstance(inputs_json, dict) and "annual_saving" in inputs_json:
            # Heuristic: presence of monthly_living_expense indicates a raw form payload
            if "monthly_living_expense" in inputs_json:
                inputs_json["annual_saving"] = float(inputs_json["annual_saving"] or 0) * 12.0
    except Exception:
        pass
    # <<< END ADD

    # ✅ Normalize BEFORE persisting so rows are always canonical
    canon = to_canonical_inputs(inputs_json)

    existing = RetirementScenario.query.filter_by(
        user_id=current_user.id, scenario_name=scenario_name
    ).first()

    if existing:
        existing.inputs_json = canon
    else:
        new_scenario = RetirementScenario(
            user_id=current_user.id,
            scenario_name=scenario_name,
            inputs_json=canon
        )
        db.session.add(new_scenario)

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

    # ✅ Convert canonical (DB) -> form field names/units for the UI
    form_inputs = canonical_to_form_inputs(scenario.inputs_json or {})

    return jsonify(
        {
            "scenario_name": scenario.scenario_name,
            "inputs_json": form_inputs,  # form-ready keys
            "created_at": scenario.created_at.isoformat(),
            "updated_at": scenario.updated_at.isoformat(),
        }
    ), 200


# === DELETE a scenario ===
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

# Whitelists for each engine
_PROJECTION_KEYS = {
    "current_age","retirement_age","annual_saving","saving_increase_rate",
    "current_assets","return_rate","return_rate_after","annual_expense",
    "cpp_monthly","cpp_start_age","cpp_end_age","asset_liquidations",
    "inflation_rate","life_expectancy","income_tax_rate"
}

def _projection_args_from_params(p):
    return {k: p[k] for k in _PROJECTION_KEYS if k in p}

def _mc_args_from_params(p):
    # MC-only params (std devs) may not be saved; default them here.
    return {
        "current_age":          p["current_age"],
        "retirement_age":       p["retirement_age"],
        "annual_saving":        p["annual_saving"],
        "saving_increase_rate": p["saving_increase_rate"],
        "current_assets":       p["current_assets"],
        "return_mean":          p["return_rate"],
        "return_mean_after":    p["return_rate_after"],
        "return_std":           p.get("return_std", 0.08),
        "annual_expense":       p["annual_expense"],
        "inflation_mean":       p["inflation_rate"],
        "inflation_std":        p.get("inflation_std", 0.005),
        "cpp_monthly":          p["cpp_monthly"],
        "cpp_start_age":        p["cpp_start_age"],
        "cpp_end_age":          p["cpp_end_age"],
        "asset_liquidations":   p.get("asset_liquidations", []),
        "life_expectancy":      p["life_expectancy"],
        "income_tax_rate":      p.get("income_tax_rate", 0.0),
        "num_simulations":      1000,
    }

# routes.py
from flask import request, jsonify, current_app
from flask_login import current_user

@projects_bp.route("/retirement/compare", methods=["POST"])
def compare_retirement():
    """
    Return JSON for compare MC (A required, B optional).
    Identical MC pipeline for both, defensive normalization, and rich errors.
    """
    def jerr(msg, code=400, extra=None):
        if extra:
            current_app.logger.warning("compare_retirement: %s | extra=%s", msg, extra)
        return jsonify({"error": msg, **({"_extra": extra} if (current_app.debug and extra) else {})}), code

    try:
        raw_a = (request.form.get("scenario_a") or "").strip()
        raw_b = (request.form.get("scenario_b") or "").strip()

        # --- parse ids
        try:
            a_id = int(raw_a)
        except Exception:
            return jerr("Invalid Scenario A id.", 400, {"scenario_a": raw_a})

        b_id = None
        if raw_b:
            try:
                b_id = int(raw_b)
            except Exception:
                return jerr("Invalid Scenario B id.", 400, {"scenario_b": raw_b})

        # --- fetch rows
        scen_a = RetirementScenario.query.get(a_id)
        if not scen_a:
            return jerr("Scenario A not found.", 404, {"a_id": a_id})
        scen_b = RetirementScenario.query.get(b_id) if b_id else None

        if current_user.is_authenticated:
            if scen_a.user_id != current_user.id:
                return jerr("You can only compare your own scenarios.", 403)
            if scen_b and scen_b.user_id != current_user.id:
                return jerr("You can only compare your own scenarios.", 403)

        # ---------- helpers ----------
        def _to_num(v):
            try:
                return float(v)
            except Exception:
                return v

        def _normalize_args(args: dict) -> dict:
            if not args:
                return {}
            args = {k: _to_num(v) for k, v in args.items()}
            # Convert percent-like inputs (7 -> 0.07) just once
            for k in (
                "return_mean", "return_std",
                "inflation_mean", "inflation_std",
                "return_rate_before", "return_rate_after",
            ):
                if k in args and isinstance(args[k], (int, float)) and args[k] >= 2:
                    args[k] = args[k] / 100.0
            # ints
            for k in ("iterations", "seed"):
                if k in args:
                    try: args[k] = int(args[k])
                    except Exception: pass
            return args

        def _run_mc_for(scn):
            params = to_canonical_inputs(scn.inputs_json or {})
            mc_args = _normalize_args(_mc_args_from_params(params))
            try:
                mc = run_monte_carlo_simulation_locked_inputs(**mc_args)
            except Exception as e:
                # bubble the exact reason up
                raise RuntimeError(f"MC failed for scenario '{scn.scenario_name}': {e}") from e

            ages = [int(x) for x in mc["ages"]]
            p10  = [float(x) for x in mc["percentiles"]["p10"]]
            p50  = [float(x) for x in mc["percentiles"]["p50"]]
            p90  = [float(x) for x in mc["percentiles"]["p90"]]
            n = min(len(ages), len(p10), len(p50), len(p90))
            return {
                "label": scn.scenario_name,
                "ages": ages[:n],
                "p10":  p10[:n],
                "p50":  p50[:n],
                "p90":  p90[:n],
                "_args": mc_args,
            }

        # ---------- A (required) ----------
        try:
            A = _run_mc_for(scen_a)
        except Exception as e:
            current_app.logger.exception("compare_retirement A failed")
            return jerr(str(e), 400)

        payload = {
            "labels": {"A": A["label"]},
            "mc": {
                "ages": A["ages"],
                "p10": {"A": A["p10"]},
                "p50": {"A": A["p50"]},
                "p90": {"A": A["p90"]},
            }
        }

        # ---------- B (optional) ----------
        if scen_b:
            try:
                B = _run_mc_for(scen_b)
            except Exception as e:
                current_app.logger.exception("compare_retirement B failed")
                return jerr(str(e), 400)

            # align A/B to same length/age
            m = min(len(A["ages"]), len(B["ages"]))
            ages = A["ages"][:m]
            payload["mc"]["ages"] = ages
            payload["mc"]["p10"]["A"] = A["p10"][:m]
            payload["mc"]["p50"]["A"] = A["p50"][:m]
            payload["mc"]["p90"]["A"] = A["p90"][:m]
            payload["mc"]["p10"]["B"] = B["p10"][:m]
            payload["mc"]["p50"]["B"] = B["p50"][:m]
            payload["mc"]["p90"]["B"] = B["p90"][:m]
            payload["labels"]["B"] = B["label"]

            # quick sanity to catch unit explosions instead of plotting nonsense
            if max(B["p50"][:m] or [0]) > 1e9:  # >$1B?
                current_app.logger.warning("Scenario B suspicious: args=%s", B["_args"])
                return jerr("Scenario B looks invalid (rates/units). Please check inputs.", 400)

        # optional debug echo
        if current_app.debug or current_app.config.get("COMPARE_DEBUG"):
            payload["_debug"] = {"A_args": A["_args"], **({"B_args": B["_args"]} if scen_b else {})}

        return jsonify(payload), 200

    except Exception as e:
        current_app.logger.exception("compare_retirement unexpected failure: %s", e)
        # show specific message when debug on
        msg = "Server error during compare."
        if current_app.debug:
            msg += f" ({e})"
        return jsonify({"error": msg}), 500


