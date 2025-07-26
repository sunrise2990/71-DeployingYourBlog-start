from flask import Blueprint, request, render_template
from models.retirement.retirement_calc import run_retirement_projection

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
    retirement_age = None

    if request.method == "POST":
        try:
            # Base inputs
            current_age = int(request.form.get("current_age"))
            retirement_age = int(request.form.get("retirement_age"))
            monthly_saving = float(request.form.get("annual_saving"))
            return_rate = float(request.form.get("return_rate")) / 100
            lifespan = int(request.form.get("lifespan"))
            monthly_living_expense = float(request.form.get("monthly_living_expense"))
            inflation_rate = float(request.form.get("inflation_rate")) / 100
            current_assets = float(request.form.get("current_assets"))

            # CPP inputs
            cpp_monthly = float(request.form.get("cpp_support") or 0)
            cpp_from = int(request.form.get("cpp_from_age") or 0)
            cpp_to = int(request.form.get("cpp_to_age") or 0)

            # Asset liquidations
            asset_liquidation = []
            for i in range(1, 4):
                amount = float(request.form.get(f"asset_liquidation_{i}") or 0)
                age = int(request.form.get(f"asset_liquidation_age_{i}") or 0)
                if amount > 0 and age > 0:
                    asset_liquidation.append({"amount": amount, "age": age})

            # Run projection
            output = run_retirement_projection(
                current_age=current_age,
                retirement_age=retirement_age,
                annual_saving=monthly_saving * 12,
                current_assets=current_assets,
                return_rate=return_rate,
                annual_expense=monthly_living_expense * 12,
                cpp_monthly=cpp_monthly,
                cpp_start_age=cpp_from,
                cpp_end_age=cpp_to,
                asset_liquidations=asset_liquidation,
                inflation_rate=inflation_rate,
                life_expectancy=lifespan,
            )

            result = output["final_assets"]
            table = [[
                row.get("Age"),
                row.get("Year"),
                "retire" if row.get("Age") == retirement_age else "",
                row.get("Living_Exp"),
                row.get("Living_Exp") if row.get("Age") >= retirement_age else "",
                row.get("Savings") if row.get("Age") < retirement_age else "",
                row.get("Asset"),
                row.get("Asset_Working") or "",
                row.get("Asset_Retirement") or "",
                row.get("Investment_Return") or "",
                row.get("Withdrawal_Rate") or "",
            ] for row in output["table"]]

            table_headers = [
                "Age", "Year", "Retire?", "Living Exp.", "Living Exp. – Ret.",
                "Savings – Before Retire", "Asset", "Asset – Working Years",
                "Asset – Retirement", "Investment Return", "Withdrawal Rate"
            ]

        except Exception as e:
            print("❌ Error in retirement projection:", e)
            result = None
            table = []
            table_headers = []

    else:
        table_headers = []

    return render_template(
        "retirement.html",
        result=result,
        table=table,
        table_headers=table_headers,
        retirement_age=retirement_age
    )
