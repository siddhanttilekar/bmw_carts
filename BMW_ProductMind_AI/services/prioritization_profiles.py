"""Prioritization profiles (Part 5).

A profile version = a set of agents, each pinned to the configuration version
current when the profile version was saved, with a weight in basis points.
Edits create new profile versions; old versions (and runs) stay intact.
Draft profiles may hold an invalid total while being edited; activation and
execution require active weights totalling exactly 100%.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.prioritization_models import PrioritizationAgent, PrioritizationProfile, ProfileAgentWeight
from services.agent_configuration import AgentConfigError, get_agent, version_snapshot
from services.agent_weights import WeightEntry, WeightValidation, bp_to_percent, percent_to_bp, validate_weights

ILLUSTRATIVE_NOTE = "Illustrative configuration - not an official BMW prioritization profile."


class ProfileError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _actor(actor: str) -> str:
    if not actor or not actor.strip():
        raise ProfileError("A user identifier is required.")
    return actor.strip()


def _write_weights(session: Session, profile: PrioritizationProfile, weights: dict[str, object], actor: str) -> None:
    if not weights:
        raise ProfileError("A profile needs at least one agent.")
    for agent_id, percent in weights.items():
        agent = get_agent(session, agent_id)
        if agent.deleted:
            raise ProfileError(f"Agent '{agent.name}' has been removed and cannot be added to a profile.")
        session.add(ProfileAgentWeight(profile_id=profile.profile_id, profile_version=profile.current_version,
                                       agent_id=agent_id, agent_version=agent.configuration_version,
                                       weight_bp=percent_to_bp(percent), created_by=actor, created_at=_now()))
    session.flush()


def create_profile(session: Session, name: str, weights: dict[str, object], created_by: str, description: str = "",
                   illustrative: bool = False) -> PrioritizationProfile:
    """``weights``: {agent_id: percent}. Saved as a draft; activate separately."""
    actor = _actor(created_by)
    if not name.strip():
        raise ProfileError("Profile name is required.")
    now = _now()
    profile = PrioritizationProfile(profile_id=f"PRF-{uuid.uuid4().hex[:10].upper()}", profile_name=name.strip(),
                                    description=(description.strip() + (f" {ILLUSTRATIVE_NOTE}" if illustrative else "")).strip(),
                                    status="draft", current_version=1, is_illustrative=illustrative, created_by=actor,
                                    created_at=now, updated_at=now)
    session.add(profile)
    session.flush()
    _write_weights(session, profile, weights, actor)
    return profile


def get_profile(session: Session, profile_id: str) -> PrioritizationProfile:
    p = session.get(PrioritizationProfile, profile_id)
    if p is None:
        raise ProfileError(f"Unknown profile {profile_id!r}")
    return p


def weights_for(session: Session, profile_id: str, version: int | None = None) -> list[ProfileAgentWeight]:
    p = get_profile(session, profile_id)
    return list(session.scalars(select(ProfileAgentWeight).where(
        ProfileAgentWeight.profile_id == profile_id, ProfileAgentWeight.profile_version == (version or p.current_version))
        .order_by(ProfileAgentWeight.weight_id)))


def validate_profile(session: Session, profile_id: str, version: int | None = None) -> WeightValidation:
    """Validates the pinned agent configurations (enabled/removed as of their
    pinned version AND currently) and the weight total."""
    entries = []
    for w in weights_for(session, profile_id, version):
        snap = version_snapshot(session, w.agent_id, w.agent_version)
        current = get_agent(session, w.agent_id)
        entries.append(WeightEntry(agent_id=f"{snap['name']} ({w.agent_id})", weight_bp=w.weight_bp,
                                   enabled=snap["enabled"] and current.enabled, deleted=snap["deleted"] or current.deleted))
    return validate_weights(entries)


def edit_profile(session: Session, profile_id: str, actor: str, *, name: str | None = None, description: str | None = None,
                 weights: dict[str, object] | None = None) -> PrioritizationProfile:
    """New profile version (re-pins every agent to its latest configuration)."""
    actor = _actor(actor)
    p = get_profile(session, profile_id)
    if p.status == "archived":
        raise ProfileError("Archived profiles cannot be edited; duplicate it instead.")
    if weights is None:
        weights = {w.agent_id: bp_to_percent(w.weight_bp) for w in weights_for(session, profile_id)}
    p.current_version += 1
    p.profile_name = (name or p.profile_name).strip()
    if description is not None:
        p.description = description.strip()
    p.updated_at = _now()
    _write_weights(session, p, weights, actor)
    if p.status == "active" and not validate_profile(session, profile_id).valid:
        p.status = "draft"  # an active profile must always be valid
    return p


def duplicate_profile(session: Session, profile_id: str, actor: str, new_name: str | None = None) -> PrioritizationProfile:
    p = get_profile(session, profile_id)
    weights = {w.agent_id: bp_to_percent(w.weight_bp) for w in weights_for(session, profile_id)}
    return create_profile(session, new_name or f"{p.profile_name} (copy)", weights, actor, p.description,
                          illustrative=False)


def activate_profile(session: Session, profile_id: str, actor: str) -> PrioritizationProfile:
    _actor(actor)
    p = get_profile(session, profile_id)
    if p.status == "archived":
        raise ProfileError("Archived profiles cannot be activated.")
    check = validate_profile(session, profile_id)
    if not check.valid:
        raise ProfileError("Cannot activate: " + " ".join(check.errors))
    for other in session.scalars(select(PrioritizationProfile).where(PrioritizationProfile.status == "active",
                                                                     PrioritizationProfile.profile_id != profile_id)):
        other.status = "inactive"
    p.status, p.updated_at = "active", _now()
    session.flush()
    return p


def archive_profile(session: Session, profile_id: str, actor: str) -> PrioritizationProfile:
    _actor(actor)
    p = get_profile(session, profile_id)
    p.status, p.updated_at = "archived", _now()
    session.flush()
    return p


class ProfileView(BaseModel):
    profile_id: str
    profile_name: str
    description: str
    status: str
    version: int
    illustrative: bool
    agents: list[dict]
    total_percent: str
    remaining_percent: str
    valid: bool
    errors: list[str]
    newer_agent_versions: list[str]
    created_by: str
    created_at: datetime
    updated_at: datetime


def profile_view(session: Session, profile_id: str, version: int | None = None) -> ProfileView:
    p = get_profile(session, profile_id)
    rows = weights_for(session, profile_id, version)
    check = validate_profile(session, profile_id, version)
    agents, newer = [], []
    for w in rows:
        snap = version_snapshot(session, w.agent_id, w.agent_version)
        current = get_agent(session, w.agent_id)
        if current.configuration_version > w.agent_version and not current.deleted:
            newer.append(snap["name"])
        agents.append({"agent_id": w.agent_id, "name": snap["name"], "agent_version": w.agent_version,
                       "scoring_method": snap["scoring_method"], "execution_method": snap["execution_method"],
                       "weight_percent": str(bp_to_percent(w.weight_bp)), "weight_bp": w.weight_bp,
                       "enabled": snap["enabled"] and current.enabled, "removed": current.deleted})
    return ProfileView(profile_id=p.profile_id, profile_name=p.profile_name, description=p.description, status=p.status,
                       version=version or p.current_version, illustrative=p.is_illustrative, agents=agents,
                       total_percent=str(check.total_percent), remaining_percent=str(check.remaining_percent),
                       valid=check.valid, errors=check.errors, newer_agent_versions=newer, created_by=p.created_by,
                       created_at=p.created_at, updated_at=p.updated_at)


def list_profiles(session: Session, include_archived: bool = True) -> list[ProfileView]:
    stmt = select(PrioritizationProfile.profile_id).order_by(PrioritizationProfile.created_at)
    if not include_archived:
        stmt = stmt.where(PrioritizationProfile.status != "archived")
    return [profile_view(session, pid) for pid in session.scalars(stmt)]


def active_profile_id(session: Session) -> str | None:
    return session.scalar(select(PrioritizationProfile.profile_id).where(PrioritizationProfile.status == "active"))


def seed_illustrative_profile(session: Session, actor: str) -> PrioritizationProfile | None:
    """Builds an illustrative profile from the starter agents' default weights
    (only enabled agents with weight > 0). Not an official BMW profile."""
    agents = list(session.scalars(select(PrioritizationAgent).where(
        PrioritizationAgent.deleted.is_(False), PrioritizationAgent.enabled.is_(True), PrioritizationAgent.default_weight_bp > 0)))
    if not agents:
        return None
    exists = session.scalar(select(func.count()).select_from(PrioritizationProfile)
                            .where(PrioritizationProfile.profile_name == "Customer Experience (illustrative)"))
    if exists:
        return None
    return create_profile(session, "Customer Experience (illustrative)",
                          {a.agent_id: bp_to_percent(a.default_weight_bp) for a in agents}, actor,
                          "Starter weighting of demand, evidence coverage, impact and feasibility.", illustrative=True)


__all__ = ["ProfileError", "AgentConfigError", "create_profile", "edit_profile", "duplicate_profile", "activate_profile",
           "archive_profile", "profile_view", "list_profiles", "validate_profile", "weights_for", "active_profile_id",
           "seed_illustrative_profile", "ILLUSTRATIVE_NOTE"]
