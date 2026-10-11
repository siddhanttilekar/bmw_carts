"""Seven-field custom agent builder (shared by the Product Prioritization
wizard and the advanced PRIORITIZE page).

Fields: Agent Name, Objective, Custom Instructions, Add Resources (incl.
uploads), Context, Weight, Execution Method. A compatible scoring rule is
shown only when the execution method offers more than one."""
from __future__ import annotations

import streamlit as st

from core.prioritization_models import EXECUTION_METHODS
from services import agent_configuration as ac
from services import agent_documents as adocs
from services import custom_metrics as cmx
from services.agent_resources import RESOURCES
from services.agent_scoring import METHODS, check_compatibility, methods_for
from ui import data

EXECUTION_LABELS = {"AI_ANALYSIS": "AI Analysis", "DATA_ANALYSIS": "Data Analysis", "HUMAN_ASSESSMENT": "Human Assessment"}


def render(prefix: str, actor: str, *, profile_to_extend=None) -> str | None:
    """Render the form; returns the new agent ID after a successful create."""
    with data.write_session() as s:
        metrics = cmx.list_custom_metrics(s)
    name = st.text_input("Agent Name", key=f"{prefix}_name")
    objective = st.text_area("Objective", height=70, key=f"{prefix}_obj",
                             placeholder="e.g. Evaluate how often customers complain about or request this capability.")
    instructions = st.text_area(
        "Custom Instructions", height=90, key=f"{prefix}_instr", max_chars=ac.MAX_INSTRUCTIONS,
        help="Scope notes for AI evidence selection (for example 'focus on touchscreen usability'). Plain text: "
             "never executed as code or SQL, and it never changes a formula.")
    resources_sel = st.multiselect("Add Resources", list(RESOURCES), format_func=lambda r: RESOURCES[r].name,
                                   key=f"{prefix}_res")
    uploads = st.file_uploader("Upload documents (PDF, XLSX, CSV, TXT, MD)", type=list(adocs.ALLOWED_TYPES),
                               accept_multiple_files=True, key=f"{prefix}_upl",
                               help="Visible to this agent only. Treated as untrusted text; never executed.")
    context = st.text_input("Context", key=f"{prefix}_ctx", max_chars=ac.MAX_CONTEXT,
                            placeholder="e.g. Next-generation BMW 5 Series")
    weight = st.number_input("Weight (%)", 0.0, 100.0, 0.0, 0.5, key=f"{prefix}_w")
    execution = st.radio("Execution Method", list(EXECUTION_METHODS), horizontal=True, key=f"{prefix}_exe",
                         format_func=lambda m: EXECUTION_LABELS.get(m, m))
    rule_ids = methods_for(execution)
    if len(rule_ids) > 1:
        rule = st.selectbox("Scoring rule", rule_ids, format_func=lambda m: METHODS[m].label, key=f"{prefix}_rule")
    else:
        rule = rule_ids[0]
    st.caption(METHODS[rule].description)
    metric_id = None
    if rule == "DOCUMENTED_CUSTOM_METRIC":
        metric_id = st.selectbox("Documented metric", [m["metric_id"] for m in metrics], key=f"{prefix}_metric",
                                 format_func=lambda m: next(f"{x['label']} ({'approved' if x['approved'] else 'NOT approved - UNSCORED'})"
                                                            for x in metrics if x["metric_id"] == m))
    final_resources = list(dict.fromkeys(resources_sel + (["RES_UPLOADED_DOCUMENTS"] if uploads else [])))
    errors, warnings = check_compatibility(execution, rule, final_resources)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.caption(w)
    add_to_profile = False
    if profile_to_extend is not None:
        add_to_profile = st.checkbox("Add to the active profile (rebalance weights to 100% afterwards)", value=True,
                                     key=f"{prefix}_add")
    if st.button("Create Agent", type="primary", key=f"{prefix}_create",
                 disabled=bool(errors) or not name.strip() or not objective.strip() or not (actor or "").strip()):
        try:
            with data.write_session() as s:
                agent, _ = ac.create_agent(s, ac.AgentSpec(
                    name=name, objective=objective, instructions=instructions.strip() or None, resources=final_resources,
                    context=context, weight_percent=weight, execution_method=execution, scoring_method=rule,
                    custom_metric_id=metric_id), actor)
                for f in uploads or []:
                    adocs.upload_document(s, agent.agent_id, f.name, f.getvalue(), actor)
                if add_to_profile:
                    from services import prioritization_profiles as pp

                    current = {a["agent_id"]: a["weight_percent"] for a in profile_to_extend.agents}
                    pp.edit_profile(s, profile_to_extend.profile_id, actor, weights={**current, agent.agent_id: weight})
                new_id = agent.agent_id
            return new_id
        except (ac.AgentConfigError, adocs.DocumentError, ValueError) as exc:
            st.error(str(exc))
    return None
