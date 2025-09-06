# models/retirement/goals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional, Union

Recurrence = Literal["once", "annual", "years"]


@dataclass
class GoalEvent:
    """
    A goal is either an extra expense (is_expense=True) or extra inflow (False).
    amount is in today's dollars; if inflation_linked=True it grows by inflation
    each year from start_age.

    recurrence:
      - "once":   applies in start_age only
      - "annual": applies every year from start_age..end_age (inclusive)
      - "years":  applies for `years` consecutive years starting at start_age

    NOTE (sign convention to match frontend & calculators):
      - Expense/outflow  -> NEGATIVE liquidation (withdrawal)
      - Inflow           -> POSITIVE liquidation (deposit)
    """
    name: str
    start_age: int
    amount: float
    is_expense: bool = True
    inflation_linked: bool = True
    recurrence: Recurrence = "once"
    end_age: Optional[int] = None   # needed for "annual"
    years: Optional[int] = None     # needed for "years"
    enabled: bool = True            # optional; ignored if False


def _inflated(base: float, years_since_start: int, inflation_rate: float, link: bool) -> float:
    if not link:
        return base
    return base * ((1 + inflation_rate) ** max(0, years_since_start))


def expand_goals_to_per_age(
    *,
    current_age: int,
    life_expectancy: int,
    inflation_rate: float,
    goals: List[Union[dict, GoalEvent]]
) -> Dict[int, Dict[str, float]]:
    """
    Returns {age: {'expense': sum_expenses, 'inflow': sum_inflows}} for ages within horizon.
    Values are POSITIVE magnitudes in their respective buckets; sign is applied later.
    """
    # normalize into GoalEvent; skip disabled items if present
    norm: List[GoalEvent] = []
    for g in goals or []:
        if isinstance(g, GoalEvent):
            if g.enabled is False:
                continue
            norm.append(g)
        else:
            if g.get("enabled") is False:
                continue
            norm.append(GoalEvent(
                name=g.get("name", ""),
                start_age=int(g["start_age"]),
                amount=float(g["amount"]),
                is_expense=bool(g.get("is_expense", True)),
                inflation_linked=bool(g.get("inflation_linked", True)),
                recurrence=g.get("recurrence", "once"),
                end_age=g.get("end_age"),
                years=g.get("years"),
            ))

    per_age: Dict[int, Dict[str, float]] = {}
    a0, a1 = int(current_age), int(life_expectancy)

    for ev in norm:
        if ev.recurrence == "once":
            ages = [ev.start_age]
        elif ev.recurrence == "annual":
            end = ev.end_age if ev.end_age is not None else a1
            ages = list(range(ev.start_age, end + 1))
        elif ev.recurrence == "years":
            n = int(ev.years or 1)
            ages = [ev.start_age + i for i in range(n)]
        else:
            ages = [ev.start_age]

        for age in ages:
            if age < a0 or age > a1:
                continue
            yrs = age - ev.start_age
            val = _inflated(float(ev.amount), yrs, float(inflation_rate), bool(ev.inflation_linked))
            bucket = "expense" if ev.is_expense else "inflow"
            per_age.setdefault(age, {"expense": 0.0, "inflow": 0.0})
            per_age[age][bucket] += float(val)

    return per_age


def goals_to_liquidations_adapter(
    *,
    current_liqs: List[dict],
    per_age: Dict[int, Dict[str, float]]
) -> List[dict]:
    """
    Convert goal cashflows into 'asset_liquidations' entries understood by the calculators.

    SIGN CONVENTION (matches frontend):
      - Expense/outflow  -> NEGATIVE liquidation (withdrawal)
      - Inflow           -> POSITIVE liquidation (deposit)

    Notes
    -----
    - This MVP maps goals post-tax and does NOT alter 'Living_Exp' for tax calcs.
    - Existing liquidations are preserved and merged by age.
    - Output is sorted by age and rounded to cents.
    """
    # Start with existing liquidations, grouped by age
    agg: Dict[int, float] = {}
    for row in (current_liqs or []):
        try:
            age = int(row.get("age"))
            amt = float(row.get("amount", 0.0))
        except Exception:
            continue
        agg[age] = agg.get(age, 0.0) + amt

    # Fold in goals: net = inflow (deposit, +) - expense (withdrawal, -)
    for age, vals in (per_age or {}).items():
        try:
            a = int(age)
        except Exception:
            continue
        inflow  = float(vals.get("inflow", 0.0))
        expense = float(vals.get("expense", 0.0))
        net = inflow - expense  # apply sign convention here
        if net:
            agg[a] = agg.get(a, 0.0) + net

    # Emit as sorted list, drop ~zero, round to cents
    out: List[dict] = []
    for age in sorted(agg.keys()):
        amt = round(agg[age], 2)
        if abs(amt) < 0.005:
            continue
        out.append({"age": int(age), "amount": float(amt)})

    return out

