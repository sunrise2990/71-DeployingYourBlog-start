from flask import Blueprint, render_template

projects_bp = Blueprint('projects', __name__, template_folder='templates')

@projects_bp.route("/projects/budget-reforecast")
def budget_reforecast():
    return render_template("budget_reforecast.html")

@projects_bp.route("/projects/budget-reforecast/leasing")
def leasing_pipeline():
    return render_template("leasing_pipeline.html")