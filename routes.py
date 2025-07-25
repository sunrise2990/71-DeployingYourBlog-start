from flask import Blueprint, request, render_template

projects_bp = Blueprint('projects', __name__, template_folder='templates')

@projects_bp.route("/projects/budget-reforecast")
def budget_reforecast():
    return render_template("budget_reforecast.html")

@projects_bp.route("/projects/budget-reforecast/leasing")
def leasing_pipeline():
    return render_template("leasing_pipeline.html")

@projects_bp.route("/projects/retirement", methods=["GET", "POST"])
def retirement():
    if request.method == "POST":
        current_age = int(request.form.get("current_age", 0))
        retire_age = int(request.form.get("retire_age", 0))
        monthly_savings = float(request.form.get("monthly_savings", 0))
        return_rate = float(request.form.get("return_rate", 0))

        return render_template("retirement.html",
                               current_age=current_age,
                               retire_age=retire_age,
                               monthly_savings=monthly_savings,
                               return_rate=return_rate)

    # On GET: Show defaults
    return render_template("retirement.html",
                           current_age=53,
                           retire_age=57,
                           monthly_savings=8000,
                           return_rate=7.0)