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

    while age <= life_expectancy:
        row = {"Age": age, "Year": year}

        # Living expense inflation adjusted
        living_expense = annual_expense * ((1 + inflation_rate) ** (age - current_age))
        row["Living_Exp"] = living_expense

        # CPP support logic
        cpp_support = 0
        if cpp_start_age and cpp_end_age and cpp_start_age <= age <= cpp_end_age:
            cpp_support = cpp_monthly * 12
        net_expense = max(living_expense - cpp_support, 0)

        # Asset liquidations
        liquidation = 0
        if asset_liquidations:
            for entry in asset_liquidations:
                if entry.get("age") == age:
                    liquidation += entry.get("amount", 0)

        # Before retirement: Save assets
        if age < retirement_age:
            savings = annual_saving
            assets += savings
            inv_return = assets * return_rate
            assets += inv_return
            row["Savings"] = savings
            row["Asset"] = assets
            row["Asset_Working"] = assets
            row["Asset_Retirement"] = None
            row["Investment_Return"] = inv_return
            row["Withdrawal_Rate"] = None
        else:
            if not asset_retirement:
                asset_retirement = assets  # Save this when retirement hits
            inv_return = assets * return_rate
            withdrawal = net_expense
            assets += inv_return - withdrawal + liquidation
            row["Savings"] = None
            row["Asset"] = assets
            row["Asset_Working"] = None
            row["Asset_Retirement"] = asset_retirement
            row["Investment_Return"] = inv_return
            row["Withdrawal_Rate"] = f"{(withdrawal / assets * 100):.1f}%" if assets > 0 else "N/A"

        result_table.append(row)
        age += 1
        year += 1

    final_assets = assets
    return {
        "retirement_age": retirement_age,
        "final_assets": final_assets,
        "table": result_table
    }
