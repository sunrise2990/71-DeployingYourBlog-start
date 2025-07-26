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
    retirement_age = None  # for displaying in template

    if request.method == "POST":
        try:
            # Collect form data
            current_age = int(request.form.get("current_age"))
            retirement_age = int(request.form.get("retirement_age"))
            annual_saving = float(request.form.get("annual_saving"))
            return_rate = float(request.form.get("return_rate")) / 100  # Convert percent to decimal

            # Placeholder values to be added later
            current_assets = 0
            annual_expense = 0

            # Run core logic
            result = run_retirement_projection(
                current_age=current_age,
                retirement_age=retirement_age,
                annual_saving=annual_saving,
                current_assets=current_assets,
                return_rate=return_rate,
                annual_expense=annual_expense
            )

        except Exception as e:
            print("❌ Error during retirement calc:", e)
            result = None

    return render_template("retirement.html", result=result, retirement_age=retirement_age)
