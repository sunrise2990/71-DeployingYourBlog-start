from flask import Blueprint, request, render_template
import numpy as np
from models.retirement.retirement_calc import (
    run_retirement_projection,
    run_monte_carlo_simulation_locked_inputs,
)

# Define blueprint
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
        "Age", "Year", "Retire?", "Living Exp.", "CPP / Extra Income", "Living Exp. – Ret.",
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
                return_rate_before = get_form_value("return_rate_before", float) / 100
                return_rate_after = get_form_value("return_rate_after", float) / 100
                lifespan = get_form_value("lifespan", int)
                monthly_living_expense = get_form_value("monthly_living_expense", float)
                inflation_rate = get_form_value("inflation_rate", float) / 100
                current_assets = get_form_value("current_assets", float)
                saving_increase_rate = get_form_value("saving_increase_rate", float) / 100

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

                # 🧠 Run Deterministic Projection
                output = run_retirement_projection(
                    current_age=current_age,
                    retirement_age=retirement_age,
                    annual_saving=monthly_saving * 12,
                    saving_increase_rate=saving_increase_rate,
                    current_assets=current_assets,
                    return_rate_before=return_rate_before,
                    return_rate_after=return_rate_after,
                    annual_expense=monthly_living_expense * 12,
                    cpp_monthly=cpp_monthly,
                    cpp_start_age=cpp_from,
                    cpp_end_age=cpp_to,
                    asset_liquidations=asset_liquidation,
                    inflation_rate=inflation_rate,
                    life_expectancy=lifespan
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
                    f"${row.get('Living_Exp_Retirement', 0):,.0f}",
                    f"${row.get('Asset_Liquidation', 0):,.0f}" if row.get("Asset_Liquidation") else "",
                    f"${row.get('Savings', 0):,.0f}" if row.get("Savings") else "",
                    f"${row.get('Asset', 0):,.0f}",
                    f"${row.get('Asset_Retirement', 0):,.0f}" if row.get("Asset_Retirement") else "",
                    f"${row.get('Investment_Return', 0):,.0f}" if row.get("Investment_Return") is not None else "",
                    f"{row.get('Effective_Return_Rate'):.1f}%" if row.get("Effective_Return_Rate") is not None else "",
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

                # 🎲 Monte Carlo Simulation
                mc_output = run_monte_carlo_simulation_locked_inputs(
                    current_age=current_age,
                    retirement_age=retirement_age,
                    annual_saving=monthly_saving * 12,
                    saving_increase_rate=saving_increase_rate,
                    current_assets=current_assets,
                    return_mean=return_rate_after,
                    return_std=return_std,
                    annual_expense=monthly_living_expense * 12,
                    inflation_mean=inflation_rate,
                    inflation_std=inflation_std,
                    cpp_monthly=cpp_monthly,
                    cpp_start_age=cpp_from,
                    cpp_end_age=cpp_to,
                    asset_liquidations=asset_liquidation,
                    life_expectancy=lifespan,
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

