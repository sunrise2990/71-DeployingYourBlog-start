def run_retirement_projection(
        current_age,
        retirement_age,
        annual_saving,
        current_assets,
        return_rate,
        annual_expense,
        cpp_monthly=0,
        cpp_start_age=None,
        cpp_end_age=None,
        asset_liquidations=None,  # list of dicts like [{"amount": 100000, "age": 60}]
        inflation_rate=0.021,
        life_expectancy=90
):
    result_table = []
    age = current_age
    year = 1
    assets = current_assets
    asset_retirement = None

    # Convert liquidations to lookup dict for quick access
    liquidation_lookup = {entry['age']: entry['amount'] for entry in asset_liquidations or []}

    while age <= life_expectancy:
        row = {
            "Age": age,
            "Year": year,
            "Retire": "retire" if age == retirement_age else "",
        }

        # Inflation-adjusted living expense
        living_expense = annual_expense * ((1 + inflation_rate) ** (age - current_age))
        row["Living_Exp"] = round(living_expense, 0)

        # CPP offset
        cpp_support = 0
        if cpp_start_age and cpp_end_age and cpp_start_age <= age <= cpp_end_age:
            cpp_support = cpp_monthly * 12
        net_expense = max(living_expense - cpp_support, 0)
        row["Living_Exp_Retirement"] = round(net_expense, 0) if age >= retirement_age else None

        # Liquidation addition
        liquidation = liquidation_lookup.get(age, 0)

        # Before retirement
        if age < retirement_age:
            savings = annual_saving
            inv_return = assets * return_rate
            assets += savings + inv_return
            row["Savings"] = round(savings, 0)
            row["Asset"] = round(assets, 0)
            row["Asset_Working"] = round(assets, 0)
            row["Asset_Retirement"] = None
            row["Investment_Return"] = round(inv_return, 0)
            row["Withdrawal_Rate"] = None
        else:
            if not asset_retirement:
                asset_retirement = assets
            inv_return = assets * return_rate
            assets += inv_return - net_expense + liquidation
            row["Savings"] = None
            row["Asset"] = round(assets, 0)
            row["Asset_Working"] = None
            row["Asset_Retirement"] = round(asset_retirement, 0)
            row["Investment_Return"] = round(inv_return, 0)
            row["Withdrawal_Rate"] = f"{(net_expense / assets * 100):.1f}%" if assets > 0 else "N/A"

        result_table.append(row)
        age += 1
        year += 1

    final_assets = assets
    return {
        "retirement_age": retirement_age,
        "final_assets": final_assets,
        "table": result_table
    }
