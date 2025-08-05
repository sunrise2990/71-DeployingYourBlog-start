from flask import Blueprint, request, render_template, jsonify
from flask_login import login_required, current_user
import numpy as np
from models.retirement.retirement_calc import (
    run_retirement_projection,
    run_monte_carlo_simulation_locked_inputs, sensitivity_analysis
)
from models import db
from models.retirement.retirement_scenario import RetirementScenario # adjust import path as needed

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

# Retirement Planner route
from flask import request, render_template
from flask_login import current_user
from models.retirement.retirement_calc import (
    run_retirement_projection,
    run_monte_carlo_simulation_locked_inputs,
    sensitivity_analysis
)
from models.retirement.retirement_scenario import RetirementScenario

@projects_bp.route("/retirement", methods=["GET", "POST"])
def retirement():
    # ——— Initialize outputs
    result = None
    table = []
    chart_data = {}
    monte_carlo_data = {}
    depletion_stats = {}
    sensitivities = {}
    reset = False
    retirement_age = None

    form_inputs = {}

    table_headers = [
        "Age", "Year", "Retire?", "Living Exp.", "CPP / Extra Income",
        "Income Tax Payment", "Living Exp. – Ret.",
        "Asset Liquidation", "Savings – Before Retire", "Asset",
        "Asset – Retirement", "Investment Return", "Return Rate", "Withdrawal Rate"
    ]

    # ——— Handle POST
    if request.method == "POST":
        action = request.form.get("action")

        if action == "reset":
            reset = True

        elif action == "calculate":
            # helper
            def get_val(name, cast, default=0):
                v = request.form.get(name)
                form_inputs[name] = v
                return cast(v) if v else default

            # parse inputs
            current_age            = get_val("current_age", int)
            retirement_age         = get_val("retirement_age", int)
            monthly_saving         = get_val("annual_saving", float)
            return_rate            = get_val("return_rate", float) / 100
            return_rate_after      = get_val("return_rate_after", float) / 100
            lifespan               = get_val("lifespan", int)
            monthly_living_expense = get_val("monthly_living_expense", float)
            inflation_rate         = get_val("inflation_rate", float) / 100
            current_assets         = get_val("current_assets", float)
            saving_increase_rate   = get_val("saving_increase_rate", float) / 100
            income_tax_rate        = get_val("income_tax_rate", float) / 100
            cpp_monthly            = get_val("cpp_support", float)
            cpp_from               = get_val("cpp_from_age", int)
            cpp_to                 = get_val("cpp_to_age", int)
            return_std             = get_val("return_std", float) / 100
            inflation_std          = get_val("inflation_std", float) / 100

            # asset liquidations
            asset_liquidation = []
            for i in range(1, 4):
                amt = get_val(f"asset_liquidation_{i}", float)
                age = get_val(f"asset_liquidation_age_{i}", int)
                if amt and age:
                    asset_liquidation.append({"amount": amt, "age": age})

            # baseline params for everything
            baseline_params = {
                "current_age":          current_age,
                "retirement_age":       retirement_age,
                "annual_saving":        monthly_saving * 12,
                "saving_increase_rate": saving_increase_rate,
                "current_assets":       current_assets,
                "return_rate":          return_rate,
                "return_rate_after":    return_rate_after,
                "annual_expense":       monthly_living_expense * 12,
                "cpp_monthly":          cpp_monthly,
                "cpp_start_age":        cpp_from,
                "cpp_end_age":          cpp_to,
                "asset_liquidations":   asset_liquidation,
                "inflation_rate":       inflation_rate,
                "life_expectancy":      lifespan,
                "income_tax_rate":      income_tax_rate
            }

            # 1) Deterministic + Monte Carlo
            try:
                out = run_retirement_projection(**baseline_params)
                result = out["final_assets"]

                # build table
                for r in out["table"]:
                    if not r.get("Living_Exp_Retirement"):
                        r["Living_Exp_Retirement"] = r["Living_Exp"]
                table = [
                    [
                        r["Age"], r["Year"], r["Retire"],
                        f"${r['Living_Exp']:,.0f}",
                        f"${r.get('CPP_Support',0):,.0f}" if r.get("CPP_Support") else "",
                        f"${r.get('Income_Tax_Payment',0):,.0f}",
                        f"${r['Living_Exp_Retirement']:,.0f}",
                        f"${r.get('Asset_Liquidation',0):,.0f}" if r.get("Asset_Liquidation") else "",
                        f"${r.get('Savings',0):,.0f}" if r.get("Savings") else "",
                        f"${r['Asset']:,.0f}",
                        f"${r.get('Asset_Retirement',0):,.0f}" if r.get("Asset_Retirement") else "",
                        f"${r.get('Investment_Return',0):,.0f}" if r.get("Investment_Return") is not None else "",
                        f"{r.get('Return_Rate',0):.1f}%" if r.get("Return_Rate") is not None else "",
                        f"{r.get('Withdrawal_Rate',0):.1f}%" if r.get("Withdrawal_Rate") is not None else ""
                    ]
                    for r in out["table"]
                ]

                chart_data = {
                    "Age": [r["Age"] for r in out["table"]],
                    "Living_Exp_Retirement": [r["Living_Exp_Retirement"] for r in out["table"]],
                    "Asset_Retirement": [r["Asset_Retirement"] for r in out["table"]],
                    "Withdrawal_Rate": [round(r.get("Withdrawal_Rate",0)/100,4) for r in out["table"]]
                }

                mc = baseline_params.copy()
                for k in ("return_rate","return_rate_after","inflation_rate"):
                    mc.pop(k, None)
                mc.update({
                    "return_mean":       return_rate,
                    "return_mean_after": return_rate_after,
                    "return_std":        return_std,
                    "inflation_mean":    inflation_rate,
                    "inflation_std":     inflation_std,
                    "num_simulations":   1000
                })
                mc_out = run_monte_carlo_simulation_locked_inputs(**mc)
                monte_carlo_data = {
                    "Age": mc_out["ages"],
                    "Percentile_10": mc_out["percentiles"]["p10"],
                    "Percentile_50": mc_out["percentiles"]["p50"],
                    "Percentile_90": mc_out["percentiles"]["p90"]
                }
                depletion_stats = {
                    "age_75": mc_out["depletion_probs"].get(75,0),
                    "age_85": mc_out["depletion_probs"].get(85,0),
                    "age_90": mc_out["depletion_probs"].get(90,0),
                    "ever":   mc_out["depletion_probs"].get("ever",0)
                }

            except Exception as e:
                print("❌ Projection/MC error:", e)
                result = None
                table = []
                chart_data = {}
                monte_carlo_data = {}
                depletion_stats = {}

            # 2) Sensitivity in its own block + filter out any None
            try:
                vars_to_test = [
                    "current_assets","return_rate","return_rate_after",
                    "annual_saving","annual_expense","saving_increase_rate",
                    "inflation_rate","income_tax_rate"
                ]
                raw_sens = sensitivity_analysis(baseline_params, vars_to_test, delta=0.01)
                sensitivities = {k: v for k, v in raw_sens.items() if v is not None}
            except Exception as se:
                print("⚠️ Sensitivity error:", se)
                sensitivities = {}

    # ——— Saved scenarios (guard for anonymous)
    selected_scenario_id = request.form.get("load_scenario_select", "")
    if current_user.is_authenticated:
        saved_scenarios = RetirementScenario.query.filter_by(user_id=current_user.id).all()
    else:
        saved_scenarios = []

    # ——— Final render
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
        sensitivities=sensitivities,
        return_std=request.form.get("return_std") or "8",
        inflation_std=request.form.get("inflation_std") or "0.5",
        selected_scenario_id=selected_scenario_id,
        saved_scenarios=saved_scenarios
    )


# ===== New Scenario Blueprint and Routes =====

scenarios_bp = Blueprint("scenarios", __name__, url_prefix="/scenarios")

@scenarios_bp.route("/save", methods=["POST"])
@login_required
def save_scenario():
    data = request.get_json()
    scenario_name = data.get("scenario_name")
    inputs_json = data.get("inputs_json")

    if not scenario_name or not inputs_json:
        return jsonify({"error": "Missing scenario_name or inputs_json"}), 400

    # Check if scenario with same name exists for user
    existing = RetirementScenario.query.filter_by(
        user_id=current_user.id, scenario_name=scenario_name
    ).first()

    if existing:
        existing.inputs_json = inputs_json
    else:
        new_scenario = RetirementScenario(
            user_id=current_user.id, scenario_name=scenario_name, inputs_json=inputs_json
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
    return jsonify(
        {
            "scenario_name": scenario.scenario_name,
            "inputs_json": scenario.inputs_json,
            "created_at": scenario.created_at.isoformat(),
            "updated_at": scenario.updated_at.isoformat(),
        }
    ), 200


# === New DELETE route to delete a scenario ===
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
