# models/retirement/retirement_calc.py

def run_retirement_projection(current_age, retire_age, lifespan, monthly_savings, starting_assets, return_rate):
    results = []
    yearly_savings = monthly_savings * 12
    age = current_age
    assets = starting_assets

    while age <= lifespan:
        is_retired = age >= retire_age
        assets *= (1 + return_rate / 100)

        if not is_retired:
            assets += yearly_savings
            savings = yearly_savings
            living_expense = 0
        else:
            savings = 0
            living_expense = 7050 * 12
            assets -= living_expense

        results.append({
            "age": age,
            "retired": is_retired,
            "assets": round(assets),
            "savings": savings,
            "living_expense": living_expense
        })

        age += 1

    return results
