from flask import Blueprint, request, render_template, jsonify
from flask_login import login_required, current_user
import numpy as np
from models.retirement.retirement_calc import (
    run_retirement_projection,
    run_monte_carlo_simulation_locked_inputs,
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
@projects_bp.route("/retirement", methods=["GET", "POST"])
def retirement():
    result = None
    table = []
    chart_data = {}
    monte_carlo_data = {}
    depletion_stats = {}
    reset = False
    retirement_age = None

    form_inputs = {}

    table_headers = [
        "Age", "Year", "Retire?", "Living Exp.", "CPP / Extra Income", "Income Tax Payment", "Living Exp. – Ret.",
        "Asset Liquidation", "Savings – Before Retire", "Asset",
        "Asset – Retirement", "Investment Return", "Return Rate", "Withdrawal Rate"
    ]

    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset":
            reset = True
        elif action == "calculate":
            try:
                def get_form_value(name, cast_func, default=0):
                    value = request.form.get(name)
                    form_inputs[name] = value
                    return cast_func(value) if value else default

                current_age = get_form_value("current_age", int)
                retirement_age = get_form_value("retirement_age", int)
                monthly_saving = get_form_value("annual_saving", float)
                return_rate = get_form_value("return_rate", float) / 100
                return_rate_after = get_form_value("return_rate_after", float) / 100
                lifespan = get_form_value("lifespan", int)
                monthly_living_expense = get_form_value("monthly_living_expense", float)
                inflation_rate = get_form_value("inflation_rate", float) / 100
                current_assets = get_form_value("current_assets", float)
                saving_increase_rate = get_form_value("saving_increase_rate", float) / 100

                income_tax_rate = get_form_value("income_tax_rate", float) / 100

                cpp_monthly = get_form_value("cpp_support", float)
                cpp_from = get_form_value("cpp_from_age", int)
                cpp_to = get_form_value("cpp_to_age", int)

                return_std = get_form_value("return_std", float) / 100
                inflation_std = get_form_value("inflation_std", float) / 100

                asset_liquidation = []
                for i in range(1, 4):
                    amt_key = f"asset_liquidation_{i}"
                    age_key = f"asset_liquidation_age_{i}"
                    amount = get_form_value(amt_key, float)
                    age = get_form_value(age_key, int)
                    if amount != 0 and age > 0:
                        asset_liquidation.append({"amount": amount, "age": age})

                output = run_retirement_projection(
                    current_age=current_age,
                    retirement_age=retirement_age,
                    annual_saving=monthly_saving * 12,
                    saving_increase_rate=saving_increase_rate,
                    current_assets=current_assets,
                    return_rate=return_rate,
                    return_rate_after=return_rate_after,
                    annual_expense=monthly_living_expense * 12,
                    cpp_monthly=cpp_monthly,
                    cpp_start_age=cpp_from,
                    cpp_end_age=cpp_to,
                    asset_liquidations=asset_liquidation,
                    inflation_rate=inflation_rate,
                    life_expectancy=lifespan,
                    income_tax_rate = income_tax_rate
                )

                result = output["final_assets"]

                for row in output["table"]:
                    if not row.get("Living_Exp_Retirement"):
                        row["Living_Exp_Retirement"] = row.get("Living_Exp", 0)

                table = [[
                    row.get("Age"),
                    row.get("Year"),
                    row.get("Retire"),
                    f"${row.get('Living_Exp', 0):,.0f}",
                    f"${row.get('CPP_Support', 0):,.0f}" if row.get("CPP_Support") else "",
                    f"${row.get('Income_Tax_Payment', 0):,.0f}",  # New column added here
                    f"${row.get('Living_Exp_Retirement', 0):,.0f}",
                    f"${row.get('Asset_Liquidation', 0):,.0f}" if row.get("Asset_Liquidation") else "",
                    f"${row.get('Savings', 0):,.0f}" if row.get("Savings") else "",
                    f"${row.get('Asset', 0):,.0f}",
                    f"${row.get('Asset_Retirement', 0):,.0f}" if row.get("Asset_Retirement") else "",
                    f"${row.get('Investment_Return', 0):,.0f}" if row.get("Investment_Return") is not None else "",
                    f"{row.get('Return_Rate'):.1f}%" if row.get("Return_Rate") is not None else "",
                    f"{row.get('Withdrawal_Rate'):.1f}%" if row.get("Withdrawal_Rate") is not None else ""
                ] for row in output["table"]]

                chart_data = {
                    "Age": [row.get("Age") for row in output["table"]],
                    "Living_Exp_Retirement": [
                        row.get("Living_Exp_Retirement") or 0 for row in output["table"]
                    ],
                    "Asset_Retirement": [
                        row.get("Asset_Retirement") if row.get("Asset_Retirement") is not None else 0
                        for row in output["table"]
                    ],
                    "Withdrawal_Rate": [
                        round(row.get("Withdrawal_Rate") / 100, 4) if row.get("Withdrawal_Rate") is not None else None
                        for row in output["table"]
                    ]
                }

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
                chart_data = {}
                monte_carlo_data = {}
                depletion_stats = {}

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
        return_std=request.form.get("return_std") or "8",
        inflation_std=request.form.get("inflation_std") or "0.5"
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
