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
    asset_liquidations=None,
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
        living_expense = annual_expense * ((1 + inflation_rate) ** (age - current_age))
        cpp_support = cpp_monthly * 12 if cpp_start_age and cpp_end_age and cpp_start_age <= age <= cpp_end_age else 0
        net_expense = max(living_expense - cpp_support, 0)

        liquidation = sum(entry["amount"] for entry in asset_liquidations or [] if entry.get("age") == age)

        row["Living_Exp"] = living_expense
        row["Living_Exp_Retirement"] = net_expense if age >= retirement_age else ""
        row["CPP_Support"] = cpp_support if age >= retirement_age else ""
        row["Asset_Liquidation"] = liquidation if age >= retirement_age else ""

        if age < retirement_age:
            savings = annual_saving
            assets += savings
            inv_return = assets * return_rate
            assets += inv_return
            row.update({
                "Savings": savings,
                "Asset": assets,
                "Asset_Working": assets,
                "Asset_Retirement": None,
                "Investment_Return": inv_return,
                "Retire": "",
                "Withdrawal_Rate": None
            })
        else:
            if not asset_retirement:
                asset_retirement = assets
            inv_return = assets * return_rate
            withdrawal = net_expense
            assets += inv_return - withdrawal + liquidation
            row.update({
                "Savings": None,
                "Asset": assets,
                "Asset_Working": None,
                "Asset_Retirement": assets,  # fix: same as Asset
                "Investment_Return": inv_return,
                "Retire": "retire" if age == retirement_age else "",
                "Withdrawal_Rate": f"{(withdrawal / assets * 100):.1f}%" if assets > 0 else "N/A"
            })

        result_table.append(row)
        age += 1
        year += 1

    return {
        "retirement_age": retirement_age,
        "final_assets": assets,
        "table": result_table
    }
