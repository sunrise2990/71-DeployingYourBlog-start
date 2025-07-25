from flask import Blueprint, request, render_template
from models.retirement.retirement_calc import run_retirement_projection


projects_bp = Blueprint('projects', __name__, template_folder='templates')

@projects_bp.route("/projects/budget-reforecast")
def budget_reforecast():
    return render_template("budget_reforecast.html")

@projects_bp.route("/projects/budget-reforecast/leasing")
def leasing_pipeline():
    return render_template("leasing_pipeline.html")



@projects_bp.route("/retirement", methods=["GET", "POST"])
def retirement():
    result = None

    if request.method == "POST":
        try:
            # Get form inputs from the frontend
            current_age = int(request.form.get("current_age"))
            retirement_age = int(request.form.get("retirement_age"))
            annual_saving = float(request.form.get("annual_saving"))
            current_assets = float(request.form.get("current_assets"))
            return_rate = float(request.form.get("return_rate")) / 100  # percent to decimal
            annual_expense = float(request.form.get("annual_expense"))

            # Call your core logic function
            result = run_retirement_projection(
                current_age=current_age,
                retirement_age=retirement_age,
                annual_saving=annual_saving,
                current_assets=current_assets,
                return_rate=return_rate,
                annual_expense=annual_expense,
            )
        except Exception as e:
            result = {"error": str(e)}

    return render_template("retirement.html", result=result)