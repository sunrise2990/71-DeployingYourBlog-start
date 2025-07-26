# retirement_calc.py

def run_retirement_projection(
    current_age,
    retirement_age,
    annual_saving,
    current_assets,
    return_rate,
    annual_expense,
    cpp_monthly,
    cpp_start_age,
    cpp_end_age,
    asset_liquidations,
    inflation_rate,
    life_expectancy,
):
    table = []
    assets = current_assets
    year = 1
    asset_retirement = None

    for age in range(current_age, life_expectancy + 1):
        row = {
            "Age": age,
            "Year": year,
            "Retire": "retire" if age == retirement_age else "",
        }

        # Calculate expenses with inflation
        inflation_factor = (1 + inflation_rate) ** (age - current_age)
        living_exp = annual_expense * inflation_factor
        row["Living_Exp"] = round(living_exp)

        # Determine if retired
        retired = age >= retirement_age

        # Net expense = living expense - CPP income (if within age)
        cpp_support = cpp_monthly * 12 if cpp_start_age <= age <= cpp_end_age else 0
        net_expense = max(0, living_exp - cpp_support)
        row["Living_Exp_Retirement"] = round(net_expense) if retired else None
        row["CPP_Support"] = round(cpp_support) if retired else None

        # Check for any asset liquidation at this age
        liquidation = sum(x["amount"] for x in asset_liquidations if x["age"] == age)
        row["Asset_Liquidation"] = round(liquidation) if retired and liquidation > 0 else None

        if not retired:
            savings = annual_saving
            inv_return = assets * return_rate
            assets += savings + inv_return

            row["Savings"] = round(savings)
            row["Asset"] = round(assets)
            row["Asset_Working"] = round(assets)
            row["Asset_Retirement"] = None
            row["Investment_Return"] = round(inv_return)
            row["Withdrawal_Rate"] = None

        else:
            if asset_retirement is None:
                asset_retirement = assets
            inv_return = assets * return_rate
            withdrawal = net_expense
            assets += inv_return - withdrawal + liquidation

            row["Savings"] = None
            row["Asset"] = round(assets)
            row["Asset_Working"] = None
            row["Asset_Retirement"] = round(asset_retirement)
            row["Investment_Return"] = round(inv_return)
            row["Withdrawal_Rate"] = f"{(withdrawal / assets * 100):.1f}%" if assets > 0 else "N/A"

        table.append(row)
        year += 1

    final_assets = assets
    return {
        "final_assets": final_assets,
        "table": table
    }
