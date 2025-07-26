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
            # 🔹 Base inputs
            current_age = int(request.form.get("current_age"))
            retirement_age = int(request.form.get("retirement_age"))
            monthly_saving = float(request.form.get("annual_saving"))
            return_rate = float(request.form.get("return_rate")) / 100
            lifespan = int(request.form.get("lifespan"))
            monthly_living_expense = float(request.form.get("monthly_living_expense"))
            inflation_rate = float(request.form.get("inflation_rate")) / 100
            current_assets = float(request.form.get("current_assets"))

            # 🔹 CPP support
            cpp_monthly = float(request.form.get("cpp_support"))
            cpp_from = int(request.form.get("cpp_from_age"))
            cpp_to = int(request.form.get("cpp_to_age"))

            # 🔹 Asset liquidation (3 slots)
            asset_liquidation_1 = float(request.form.get("asset_liquidation_1") or 0)
            asset_liquidation_age_1 = int(request.form.get("asset_liquidation_age_1") or 0)

            asset_liquidation_2 = float(request.form.get("asset_liquidation_2") or 0)
            asset_liquidation_age_2 = int(request.form.get("asset_liquidation_age_2") or 0)

            asset_liquidation_3 = float(request.form.get("asset_liquidation_3") or 0)
            asset_liquidation_age_3 = int(request.form.get("asset_liquidation_age_3") or 0)

            asset_liquidation = []
            if asset_liquidation_1 > 0:
                asset_liquidation.append((asset_liquidation_age_1, asset_liquidation_1))
            if asset_liquidation_2 > 0:
                asset_liquidation.append((asset_liquidation_age_2, asset_liquidation_2))
            if asset_liquidation_3 > 0:
                asset_liquidation.append((asset_liquidation_age_3, asset_liquidation_3))

            # 🧠 Run simulation
            result, table = run_retirement_projection(
                current_age=current_age,
                retirement_age=retirement_age,
                annual_saving=monthly_saving * 12,
                return_rate=return_rate,
                current_assets=current_assets,
                lifespan=lifespan,
                monthly_living_expense=monthly_living_expense,
                inflation_rate=inflation_rate,
                cpp_support=cpp_monthly,
                cpp_from=cpp_from,
                cpp_to=cpp_to,
                asset_liquidation=asset_liquidation
            )

        except Exception as e:
            print("❌ Error in retirement projection:", e)
            result = None
            table = []

    return render_template(
        "retirement.html",
        result=result,
        table=table,
        retirement_age=retirement_age
    )
