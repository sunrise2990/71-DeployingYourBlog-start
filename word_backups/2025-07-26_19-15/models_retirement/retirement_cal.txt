def run_retirement_projection(
    current_age,
    retirement_age,
    annual_saving,
    saving_increase_rate,
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

    for age in range(current_age, life_expectancy + 1):
        row = {
            "Age": age,
            "Year": year,
            "Retire": "retire" if age == retirement_age else "",
        }

        # 🔸 Inflation-adjusted expense
        inflation_factor = (1 + inflation_rate) ** (age - current_age)
        living_exp = annual_expense * inflation_factor
        row["Living_Exp"] = round(living_exp)

        # 🔸 CPP Support
        cpp_support = cpp_monthly * 12 if cpp_start_age <= age <= cpp_end_age else 0
        row["CPP_Support"] = round(cpp_support) if cpp_support != 0 else None

        # 🔸 Net retirement expense
        retired = age >= retirement_age
        net_expense = living_exp - cpp_support
        row["Living_Exp_Retirement"] = round(net_expense) if retired else None

        # 🔸 Asset Liquidation
        liquidation = sum(x["amount"] for x in asset_liquidations if x["age"] == age)
        row["Asset_Liquidation"] = round(liquidation) if liquidation != 0 else None

        if not retired:
            # 🔹 While working
            saving_factor = (1 + saving_increase_rate) ** (age - current_age)
            savings = annual_saving * saving_factor
            inv_return = assets * return_rate
            assets += savings + inv_return

            row["Savings"] = round(savings)
            row["Asset"] = round(assets)
            row["Asset_Retirement"] = round(assets)
            row["Investment_Return"] = return_rate  # raw rate like 0.055
            row["Withdrawal_Rate"] = None

        else:
            # 🔹 Retirement phase
            inv_return = assets * return_rate
            withdrawal = net_expense
            assets += inv_return - withdrawal + liquidation

            row["Savings"] = None
            row["Asset"] = round(assets)
            row["Asset_Retirement"] = round(assets)
            row["Investment_Return"] = return_rate  # raw rate like 0.056
            row["Withdrawal_Rate"] = (withdrawal / assets * 100) if assets > 0 else None

        table.append(row)
        year += 1

    return {
        "final_assets": assets,
        "table": table
    }

