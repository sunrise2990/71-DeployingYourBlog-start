import numpy as np

# 🔹 Deterministic Retirement Projection
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
            saving_factor = (1 + saving_increase_rate) ** (age - current_age)
            savings = annual_saving * saving_factor
            inv_return = assets * return_rate
            assets += savings + inv_return

            row["Savings"] = round(savings)
            row["Asset"] = round(assets)
            row["Asset_Retirement"] = round(assets)
            row["Investment_Return"] = round(inv_return)
            row["Withdrawal_Rate"] = None

        else:
            inv_return = assets * return_rate
            withdrawal = net_expense
            assets += inv_return - withdrawal + liquidation

            row["Savings"] = None
            row["Asset"] = round(assets)
            row["Asset_Retirement"] = round(assets)
            row["Investment_Return"] = round(inv_return)
            row["Withdrawal_Rate"] = round((withdrawal / assets * 100), 1) if assets > 0 else None

        table.append(row)
        year += 1

    return {
        "final_assets": round(assets),
        "table": table
    }


# 🔹 Monte Carlo Simulation with realistic assumptions
def run_monte_carlo_simulation_locked_inputs(
    *,
    current_age: int,
    retirement_age: int,
    annual_saving: float,
    saving_increase_rate: float,
    current_assets: float,
    return_mean: float,
    return_std: float,
    annual_expense: float,
    inflation_mean: float,
    inflation_std: float,
    cpp_monthly: float,
    cpp_start_age: int,
    cpp_end_age: int,
    asset_liquidations: list,
    life_expectancy: int,
    num_simulations: int = 1000,
):

    years = life_expectancy - current_age + 1
    ages = np.arange(current_age, life_expectancy + 1)
    sim_paths = np.zeros((num_simulations, years), dtype=float)

    for s in range(num_simulations):
        assets = current_assets
        cum_infl = 1.0

        for idx, age in enumerate(ages):
            rand_return = np.random.normal(return_mean, return_std)  # 🔸 keep nominal
            rand_infl = np.random.normal(inflation_mean, inflation_std)
            cum_infl *= (1 + rand_infl)

            living_exp = annual_expense * cum_infl
            cpp_support = cpp_monthly * 12 if cpp_start_age <= age <= cpp_end_age else 0.0
            liquidation = sum(x["amount"] for x in asset_liquidations if x["age"] == age)
            retired = age >= retirement_age

            if not retired:
                saving_factor = (1 + saving_increase_rate) ** (age - current_age)
                savings = annual_saving * saving_factor
                inv_return = assets * rand_return
                assets = max(0.0, assets + savings + inv_return)
            else:
                withdrawal = max(0.0, living_exp - cpp_support)
                inv_return = assets * rand_return
                assets = max(0.0, assets + inv_return - withdrawal + liquidation)

            sim_paths[s, idx] = assets

    p10 = np.percentile(sim_paths, 10, axis=0).round(0)
    p50 = np.percentile(sim_paths, 50, axis=0).round(0)
    p90 = np.percentile(sim_paths, 90, axis=0).round(0)

    probs = _compute_depletion_probabilities(sim_paths, current_age, [75, 85, 90])

    return {
        "ages": ages.tolist(),
        "sim_paths": sim_paths,
        "percentiles": {
            "p10": p10.tolist(),
            "p50": p50.tolist(),
            "p90": p90.tolist(),
        },
        "depletion_probs": probs,
    }


# 🔸 Track % of simulations depleted before checkpoints
def _compute_depletion_probabilities(sim_paths: np.ndarray, start_age: int, checkpoints: list[int]):
    n_sims, n_years = sim_paths.shape
    probs = {}
    ever_zero = (sim_paths == 0).any(axis=1).mean()

    for cp_age in checkpoints:
        idx = cp_age - start_age
        if idx < 0:
            probs[cp_age] = 0.0
            continue
        if idx >= n_years:
            idx = n_years - 1
        depleted = (sim_paths[:, : idx + 1] == 0).any(axis=1).mean()
        probs[cp_age] = depleted

    probs["ever"] = ever_zero
    return probs

