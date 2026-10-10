"""
BMW PM Intelligence & Requirements Co-Pilot - FastAPI Backend
Provides RESTful endpoints for:
1. Regional signals, verbatims summary, customer studies & competitor benchmarks
2. SMART requirements querying (filtering by region, domain, decision status)
3. Dynamic prioritization scoring with custom weights
4. PM decision review state machine (Approve, Reject, Challenge, Modify) with mandatory audit trail
5. G0/G1 Concept Brief formal export
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import backend.requirements_engine as req_engine

app = FastAPI(
    title="BMW PM Intelligence & Requirements Co-Pilot API",
    description="Automated signal extraction, SMART requirement definition, regional prioritization, and PM audit trail for Next-Gen BMW 5 Series / i5",
    version="1.0.0"
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = r"C:\EHL Munich\data"
SIGNALS_FILE = os.path.join(DATA_DIR, "processed_signals.json")

def load_signals_data() -> Dict[str, Any]:
    if not os.path.exists(SIGNALS_FILE):
        import backend.data_engine as data_eng
        return data_eng.extract_signals()
    with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Pydantic Models for Input Validation
class PrioritizeRequest(BaseModel):
    region: str = "Global"
    weight_demand: float = Field(0.35, ge=0.0, le=1.0)
    weight_urgency: float = Field(0.30, ge=0.0, le=1.0)
    weight_brand: float = Field(0.25, ge=0.0, le=1.0)
    weight_effort: float = Field(0.10, ge=0.0, le=1.0)

class DecisionRequest(BaseModel):
    requirement_id: str
    actor: str = "Product_Manager"
    action: str = Field(..., description="APPROVED | REJECTED | CHALLENGED | MODIFIED")
    rationale: str = Field(..., min_length=5, description="Mandatory PM business rationale")
    updated_title: Optional[str] = None
    updated_smart_specification: Optional[str] = None
    updated_acceptance_criteria: Optional[List[str]] = None

# --- API Endpoints ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "vehicle_line": "Next-Generation BMW 5 Series / i5 (G60 / G68 Successor)",
        "platform": "BMW Neue Klasse (800V Architecture)"
    }

@app.get("/api/signals")
def get_signals(region: Optional[str] = Query(None, description="China | Europe | USA | Global")):
    data = load_signals_data()
    regional_signals = data.get("regional_signals", {})
    
    if region and region != "Global":
        filtered_cluster = regional_signals.get(region)
        if not filtered_cluster:
            raise HTTPException(status_code=404, detail=f"Region '{region}' not found")
        return {
            "metadata": data.get("metadata"),
            "region": region,
            "signal_cluster": filtered_cluster,
            "competitors": data.get("competitor_benchmarks", {}).get(region, []),
            "annual_report": data.get("metadata", {}).get("annual_report", {})
        }
        
    return {
        "metadata": data.get("metadata"),
        "region": "Global",
        "regional_signals": regional_signals,
        "competitor_benchmarks": data.get("competitor_benchmarks", {}),
        "sales_forecast": data.get("sales_forecast", {}),
        "annual_report": data.get("metadata", {}).get("annual_report", {})
    }

@app.get("/api/requirements")
def list_requirements(
    region: Optional[str] = Query(None, description="China | Europe | USA | Global"),
    domain: Optional[str] = Query(None, description="Interior | Exterior | Powertrain | Digital_ADAS"),
    status: Optional[str] = Query(None, description="DRAFT | APPROVED | REJECTED | CHALLENGED | MODIFIED"),
    search: Optional[str] = Query(None, description="Free text keyword search")
):
    db = req_engine.load_requirements_db()
    reqs = db.get("requirements", [])
    
    # Filter by region
    if region and region != "Global":
        reqs = [r for r in reqs if r.get("target_region") == region or r.get("target_region") == "Global"]
        
    # Filter by domain
    if domain:
        reqs = [r for r in reqs if r.get("domain", "").lower() == domain.lower()]
        
    # Filter by decision status
    if status:
        reqs = [r for r in reqs if r.get("decision_status", "").upper() == status.upper()]
        
    # Filter by search keyword
    if search:
        s_lower = search.lower()
        reqs = [
            r for r in reqs 
            if s_lower in r.get("title", "").lower() 
            or s_lower in r.get("smart_specification", "").lower() 
            or s_lower in r.get("requirement_id", "").lower()
        ]
        
    # Sort by priority composite score descending
    reqs.sort(key=lambda x: x.get("prioritization_score", {}).get("composite_score", 0), reverse=True)
    
    return {
        "total_count": len(reqs),
        "active_region": region or "Global",
        "requirements": reqs
    }

@app.get("/api/requirements/{requirement_id}")
def get_requirement(requirement_id: str):
    db = req_engine.load_requirements_db()
    for r in db.get("requirements", []):
        if r.get("requirement_id") == requirement_id:
            return r
    raise HTTPException(status_code=404, detail=f"Requirement '{requirement_id}' not found")

@app.post("/api/prioritize")
def recalculate_prioritization(payload: PrioritizeRequest):
    db = req_engine.load_requirements_db()
    custom_weights = {
        "weight_demand": payload.weight_demand,
        "weight_urgency": payload.weight_urgency,
        "weight_brand": payload.weight_brand,
        "weight_effort": payload.weight_effort,
        "description": f"Custom sensitivity profile for {payload.region}"
    }
    
    updated_reqs = []
    for r in db.get("requirements", []):
        scoring_inputs = r.get("scoring_inputs", {})
        demand = scoring_inputs.get("customer_demand", 8.0)
        urgency = scoring_inputs.get("competitive_urgency", 8.0)
        brand = scoring_inputs.get("brand_alignment", 8.0)
        effort = scoring_inputs.get("effort_score", 5.0)
        
        raw_score = (
            (demand * payload.weight_demand) +
            (urgency * payload.weight_urgency) +
            (brand * payload.weight_brand) -
            (effort * payload.weight_effort)
        )
        composite = round(max(1.0, min(10.0, raw_score)), 1)
        
        score_info = {
            "composite_score": composite,
            "customer_demand": demand,
            "competitive_urgency": urgency,
            "brand_alignment": brand,
            "effort_proxy": "HIGH" if effort >= 7.5 else ("MEDIUM" if effort >= 5.0 else "LOW"),
            "effort_score": effort,
            "active_region_profile": payload.region,
            "is_custom_weight": True
        }
        r_copy = dict(r)
        r_copy["prioritization_score"] = score_info
        updated_reqs.append(r_copy)
        
    updated_reqs.sort(key=lambda x: x["prioritization_score"]["composite_score"], reverse=True)
    
    return {
        "region": payload.region,
        "applied_weights": custom_weights,
        "requirements": updated_reqs
    }

@app.post("/api/decisions")
def record_pm_decision(payload: DecisionRequest):
    action = payload.action.upper()
    if action not in ["APPROVED", "REJECTED", "CHALLENGED", "MODIFIED"]:
        raise HTTPException(status_code=400, detail="Action must be APPROVED, REJECTED, CHALLENGED, or MODIFIED")
        
    db = req_engine.load_requirements_db()
    target_req = None
    
    for r in db.get("requirements", []):
        if r.get("requirement_id") == payload.requirement_id:
            target_req = r
            break
            
    if not target_req:
        raise HTTPException(status_code=404, detail=f"Requirement '{payload.requirement_id}' not found")
        
    old_status = target_req.get("decision_status", "DRAFT")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    audit_entry = {
        "timestamp": timestamp,
        "actor": payload.actor,
        "action": action,
        "previous_status": old_status,
        "rationale": payload.rationale
    }
    
    target_req["decision_status"] = action
    
    if payload.updated_title:
        audit_entry["title_updated"] = f"{target_req['title']} -> {payload.updated_title}"
        target_req["title"] = payload.updated_title
        
    if payload.updated_smart_specification:
        audit_entry["specification_updated"] = True
        target_req["smart_specification"] = payload.updated_smart_specification
        
    if payload.updated_acceptance_criteria:
        audit_entry["criteria_updated"] = True
        target_req["acceptance_criteria"] = payload.updated_acceptance_criteria
        
    target_req.setdefault("audit_trail", []).append(audit_entry)
    
    # Save back to persistence
    req_engine.save_requirements_db(db)
    
    return {
        "message": f"Requirement {payload.requirement_id} successfully updated to {action}",
        "requirement": target_req,
        "audit_entry": audit_entry
    }

@app.get("/api/audit-trail")
def get_global_audit_trail():
    db = req_engine.load_requirements_db()
    global_trail = []
    
    for r in db.get("requirements", []):
        for entry in r.get("audit_trail", []):
            item = dict(entry)
            item["requirement_id"] = r.get("requirement_id")
            item["requirement_title"] = r.get("title")
            item["target_region"] = r.get("target_region")
            item["domain"] = r.get("domain")
            global_trail.append(item)
            
    # Sort chronological descending
    global_trail.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {
        "total_records": len(global_trail),
        "audit_trail": global_trail
    }

@app.get("/api/export")
def export_g0_concept_brief(format: str = Query("json", description="json | markdown")):
    db = req_engine.load_requirements_db()
    reqs = db.get("requirements", [])
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    if format.lower() == "markdown":
        md = []
        md.append("# BMW AG — Concept Freeze & Project Approval Brief (Gate G0 / G1)")
        md.append(f"**Vehicle Line:** Next-Generation BMW 5 Series / i5 (G60 / G68 Successor)")
        md.append(f"**Platform Architecture:** BMW Neue Klasse (800V Architecture)")
        md.append(f"**Generated Date:** {timestamp}")
        md.append(f"**Status:** Executive Product Management Review\n")
        md.append("---\n")
        
        md.append("## 1. Executive Summary & Market Context")
        md.append("- **Strategic Imperative:** Defend executive sedan leadership in China against domestic premium NEVs (NIO, Li Auto, XPENG) while reinforcing European Autobahn acoustic and driving dynamics.")
        md.append("- **Annual Report Reference:** Grounded in BMW Group 2025 deliveries (299k 5-Series units) and €8,319M R&D commitment to Neue Klasse and localized Chinese software (~70% local OS X code with DeepSeek & Alibaba).\n")
        
        md.append("## 2. Derived SMART Product Requirements")
        for idx, r in enumerate(reqs, 1):
            score = r.get("prioritization_score", {})
            evidence = r.get("evidence", {})
            md.append(f"### {idx}. [{r.get('requirement_id')}] {r.get('title')}")
            md.append(f"- **Region:** {r.get('target_region')} | **Domain:** {r.get('domain')} | **Status:** **{r.get('decision_status')}**")
            md.append(f"- **Composite Priority Score:** {score.get('composite_score')}/10 (Demand: {score.get('customer_demand')}, Urgency: {score.get('competitive_urgency')}, Effort: {score.get('effort_proxy')})")
            md.append(f"- **SMART Specification:** {r.get('smart_specification')}")
            md.append(f"- **Acceptance Criteria:**")
            for crit in r.get("acceptance_criteria", []):
                md.append(f"  * {crit}")
            md.append(f"- **Evidence Lineage (Confidence: {evidence.get('level')}):**")
            for src in evidence.get("sources", []):
                md.append(f"  * [{src.get('source_type')}] {src.get('citation')}")
            md.append(f"- **Assumptions & Uncertainties:**")
            for asm in r.get("assumptions_uncertainties", []):
                md.append(f"  * {asm}")
            md.append("")
            
        md_text = "\n".join(md)
        return {"format": "markdown", "content": md_text}
        
    return {
        "brief_metadata": {
            "title": "BMW Next-Gen 5 Series (G60/G68) Gate G0/G1 Concept Brief",
            "date": timestamp,
            "vehicle_line": "G60/G68 Successor",
            "platform": "Neue Klasse 800V",
            "total_requirements": len(reqs),
            "approved_count": len([r for r in reqs if r.get("decision_status") == "APPROVED"])
        },
        "requirements": reqs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
