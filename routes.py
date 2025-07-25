from flask import Blueprint, render_template, request
from flask_login import login_required

projects_bp = Blueprint('projects', __name__, template_folder='templates')

@projects_bp.route("/projects/budget-reforecast")
def budget_reforecast():
    return render_template("budget_reforecast.html")

@projects_bp.route("/projects/budget-reforecast/leasing")
def leasing_pipeline():
    return render_template("leasing_pipeline.html")


# ✅ NEW: Retirement Planner Route (plug into existing blueprint)
@projects_bp.route("/projects/retirement", methods=["GET", "POST"])
def retirement():
    # These values would normally come from calculations or a database
    inputs = {
        "current_age": 53,
        "retire_age": 57,
        "monthly_savings": 8000,
        "return_rate": 7.37,
    }

    savings_table = [
        {"age": 57, "monthly": 8000, "savings": 53, "rate": 53},
        {"age": 58, "monthly": 5056, "savings": 34, "rate": 34},
        {"age": 59, "monthly": 3106, "savings": 21, "rate": 21},
    ]

    total_assets = 1684870
    legacy_amount = 1684870
    delta = 0
    surplus = 1684870
    withdrawal_rate = 7.37
    depletion_age = None
    message = "You can retire at the end of age 57 with monthly savings of $8,000"

    return render_template(
        "retirement.html",
        inputs=inputs,
        savings_table=savings_table,
        total_assets=total_assets,
        legacy_amount=legacy_amount,
        delta=delta,
        surplus=surplus,
        withdrawal_rate=withdrawal_rate,
        depletion_age=depletion_age,
        message=message,
    )
