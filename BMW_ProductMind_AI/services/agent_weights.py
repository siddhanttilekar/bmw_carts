"""Weight handling in integer basis points (10000 bp = 100.00 %).

Percent inputs are parsed with Decimal (max. 2 decimals) so there are no
floating-point sums. Validation reports the remaining weight; nothing is
rebalanced automatically - ``distribute_remaining`` only returns a proposal.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from pydantic import BaseModel

from core.prioritization_models import BP_TOTAL


class WeightError(ValueError):
    pass


def percent_to_bp(value) -> int:
    try:
        d = Decimal(str(value).strip().rstrip("%") or "0")
    except InvalidOperation as exc:
        raise WeightError(f"Weight '{value}' is not a number.") from exc
    if d < 0:
        raise WeightError("Weights must be nonnegative.")
    if d > 100:
        raise WeightError("A weight cannot exceed 100%.")
    if d != d.quantize(Decimal("0.01")):
        raise WeightError("Weights may have at most two decimals.")
    return int(d * 100)


def bp_to_percent(bp: int) -> Decimal:
    return (Decimal(bp) / 100).quantize(Decimal("0.01"))


class WeightEntry(BaseModel):
    agent_id: str
    weight_bp: int
    enabled: bool = True
    deleted: bool = False


class WeightValidation(BaseModel):
    valid: bool
    total_bp: int
    remaining_bp: int
    active_agents: int
    errors: list[str]

    @property
    def total_percent(self) -> Decimal:
        return bp_to_percent(self.total_bp)

    @property
    def remaining_percent(self) -> Decimal:
        return bp_to_percent(self.remaining_bp)


def validate_weights(entries: list[WeightEntry]) -> WeightValidation:
    errors = []
    for e in entries:
        if e.weight_bp < 0:
            errors.append(f"{e.agent_id}: weight is negative.")
        if e.weight_bp > BP_TOTAL:
            errors.append(f"{e.agent_id}: weight exceeds 100%.")
        if e.deleted:
            errors.append(f"{e.agent_id}: removed agents cannot be part of a profile.")
        elif not e.enabled:
            errors.append(f"{e.agent_id}: inactive agents cannot contribute; remove it from the profile or activate it.")
    active = [e for e in entries if e.enabled and not e.deleted]
    total = sum(e.weight_bp for e in active)
    if not active:
        errors.append("At least one active agent is required.")
    if total != BP_TOTAL:
        errors.append(f"Active weights total {bp_to_percent(total)}%, not 100%.")
    return WeightValidation(valid=not errors, total_bp=total, remaining_bp=BP_TOTAL - total, active_agents=len(active),
                            errors=errors)


def distribute_remaining(weights: dict[str, int], among: list[str]) -> dict[str, int]:
    """PROPOSAL only: spread the remaining basis points evenly over ``among``
    (remainder to the first entries). The caller must show it before applying."""
    remaining = BP_TOTAL - sum(weights.values())
    if remaining <= 0 or not among:
        return dict(weights)
    share, extra = divmod(remaining, len(among))
    proposal = dict(weights)
    for i, agent_id in enumerate(among):
        proposal[agent_id] = proposal.get(agent_id, 0) + share + (1 if i < extra else 0)
    return proposal


def weighted_contribution(score: float, weight_bp: int) -> Decimal:
    """score * weight / 100, exact to 4 decimals."""
    return (Decimal(str(score)) * Decimal(weight_bp) / Decimal(BP_TOTAL)).quantize(Decimal("0.0001"), ROUND_HALF_UP)
