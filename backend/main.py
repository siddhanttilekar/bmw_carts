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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import backend.requirements_engine as req_engine

app = FastAPI(
    title="BMW PM Intelligence & Requirements Co-Pilot API",
    description="Automated signal extraction, SMART requirement definition, regional prioritization, and PM audit trail for Next-Gen BMW 5 Series / i5",
    version="1.0.0"
)

FRONTEND_DIR = r"C:\EHL Munich\frontend"

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "BMW PM Co-Pilot API is running. Access /docs for API documentation."}


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
def get_signals(
    region: Optional[str] = Query("China", description="China | Europe | USA | Global"),
    vehicle_series: Optional[str] = Query("5 Series", description="5 Series | 1 Series | 7 Series | All")
):
    data = load_signals_data()
    regional_signals = data.get("regional_signals", {})
    all_competitors = data.get("competitor_benchmarks", {})
    sales = data.get("sales_forecast", {})
    
    target_region = region if region in ["China", "Europe", "USA", "Global"] else "China"
    active_series = "5 Series" if (not vehicle_series or vehicle_series == "All") else vehicle_series
    series_comps = all_competitors.get("5 Series", {})
    
    # Regional Delivery Metrics for BMW 5 Series / i5
    delivery_metrics_map = {
        "China": {
            "units": "68,400 units",
            "sub": "G68 Long-Wheelbase (Dadong Plant)",
            "share": "22.8% of Global 5 Series"
        },
        "Europe": {
            "units": "114,800 units",
            "sub": "G60 Saloon & Touring (+7.3% YoY)",
            "share": "38.3% of Global 5 Series"
        },
        "USA": {
            "units": "74,500 units",
            "sub": "G60 Sedan (+5.6% Americas YoY)",
            "share": "24.9% of Global 5 Series"
        },
        "Global": {
            "units": "299,582 units",
            "sub": "+19.5% YoY (Dingolfing & Dadong)",
            "share": "13.8% of Total BMW Group"
        }
    }
    
    # Regional Market Context for BMW 5 Series / i5
    market_context_map = {
        "China": "BMW 5 Series / i5 (G68 Long-Wheelbase) is BMW's primary executive profit generator in China (68k+ units in 2025). Executive customer expectations for chauffeur rear luxury clash with fierce competition from domestic premium NEVs (Li Auto L8, NIO ET7, XPENG G9, Xiaomi SU7). Critical customer demand highlights rear console compressor refrigeration (8.5L), rapid seat cooling/massage, and sub-300ms localized voice AI powered by DeepSeek and Alibaba.",
        "Europe": "BMW 5 Series / i5 (G60) is the dominant executive business and corporate fleet saloon in Europe (114k+ units in 2025, +7.3% YoY). Core customer priorities mandate whisper-quiet Autobahn acoustics (≤61 dB(A) at 130 km/h), intuitive tactile HVAC controls meeting Euro NCAP 2026 guidelines without screen distractions, and real-world winter highway range preservation.",
        "USA": "BMW 5 Series / i5 (G60) serves executive interstate commuters in North America (74k+ units in 2025). The top customer pain point is non-Tesla charging unreliability, demanding factory-standard NACS charging port with instant Plug & Charge, along with American console storage clearance for 36oz insulated travel tumblers and wireless phone cooling.",
        "Global": "Next-Generation BMW 5 Series / i5 (G60 / G68 Successor) represents BMW's global executive benchmark (299,582 units delivered in 2025, +19.5% YoY, 13.8% of total BMW Group volume). Built on the Neue Klasse 800V silicon-carbide platform, balancing sheer driving pleasure, central domain compute, and regional customization modules."
    }
    
    # Grounded Annual Report 2025 Anchors
    annual_report_map = {
        "China": "Annual Report 2025 (p. 29, 79): BMW Group delivered 626,000 units in China. BMW OS X Chinese derivative incorporates ~70% localized software architecture; R&D partnerships established with DeepSeek and Alibaba for in-cabin conversational AI and Amap lane-level navigation on BMW Panoramic Vision.",
        "Europe": "Annual Report 2025 (p. 25, 53, 79): European deliveries grew +7.3% YoY to 1,017,600 units. 5 Series G60 remains the backbone of executive mobility. Capital R&D allocation (€8,319M) deployed into Neue Klasse 800V architecture and next-gen thermal heat pump technology to protect winter efficiency.",
        "USA": "Annual Report 2025 (p. 54, 79): Americas deliveries expanded +5.6% YoY. UN Regulation 171 approved Highway Assistant operating up to 130 km/h with eye-activated lane change. Standardized adoption of NACS high-power charging confirmed for North American Neue Klasse portfolio.",
        "Global": "Annual Report 2025 (p. 25, 79): 5 Series reached 299,582 deliveries (+19.5% YoY, representing 13.8% of all BMW Group vehicles). R&D expenditure reached €8,319 Million, driving Neue Klasse 800V battery development and BMW Panoramic Vision head-up display system."
    }
    
    # Honest AI Conflict & Uncertainty Cards by Region
    conflicts_map = {
        "China": [
            {
                "type": "CONFLICT",
                "title": "Chauffeur Rear Luxury (8.5L Fridge) vs Curb Weight & Trunk Space",
                "description": "Chinese executive buyers expect an active compressor mini-fridge (8.5L) and luxury recline seating, which adds ~14kg and consumes center trunk pass-through.",
                "resolution_strategy": "Engineered integrated modular console housing lightweight compressor preserving 480L trunk capacity and golf bag pass-through."
            },
            {
                "type": "CONFLICT",
                "title": "Localized Digital Ecosystem (WeChat / DeepSeek) vs Munich Centralized OS",
                "description": "Chinese customers demand deep integration with domestic tech giants, while Munich engineering prioritizes global Linux-based BMW OS X consistency.",
                "resolution_strategy": "Architecture split running ~70% localized software stack with DeepSeek edge/cloud LLM running within the localized Chinese telemetry domain."
            },
            {
                "type": "UNCERTAINTY",
                "title": "Chauffeur vs Owner-Driver Ratio Evolution in Tier 2/3 Chinese Cities",
                "description": "High chauffeur usage in Beijing/Shanghai vs owner-driven younger buyers in Chengdu/Shenzhen.",
                "resolution_strategy": "Standardize executive comfort features in high-trim variants while maintaining rear fold-down functionality."
            }
        ],
        "Europe": [
            {
                "type": "CONFLICT",
                "title": "Euro NCAP 2026 Physical Button Safety Protocol vs Modern Glass Cockpit Minimalism",
                "description": "Euro NCAP guidelines penalize vehicles forcing critical functions (defrost, hazard, climate, wipers) into touchscreen sub-menus, while Neue Klasse design champions minimalist panoramic surfaces.",
                "resolution_strategy": "Dedicated tactile haptic quick-access bar with <0.5s eyes-off-road confirmation beneath the Panoramic Vision display and rotary steering controllers."
            },
            {
                "type": "CONFLICT",
                "title": "Autobahn High-Speed Acoustic Refinement (160+ km/h) vs Vehicle Curb Weight",
                "description": "Acoustic laminated double-pane glazing and underbody aerodynamic fairings add ~4.5 kg, penalizing electric consumption metrics.",
                "resolution_strategy": "Unbundled acoustic glass standard on G60; weight offset through Neue Klasse structural battery housing."
            },
            {
                "type": "UNCERTAINTY",
                "title": "Fleet Electrification Adoption Pace across Central vs Southern Europe",
                "description": "Nordic/German corporate fleets demand pure BEV (i5), while Southern European managers still require PHEV flexibility.",
                "resolution_strategy": "Flexible Neue Klasse multi-energy assembly line at Plant Dingolfing supporting both pure BEV and ultra-long-range PHEV."
            }
        ],
        "USA": [
            {
                "type": "CONFLICT",
                "title": "American 36oz Insulated Tumbler Sizing vs European Center Console Width",
                "description": "US drivers demand massive cup holders (98mm base) that conflict with the narrow ergonomic center tunnel design preferred by German vehicle dynamics engineers.",
                "resolution_strategy": "Neue Klasse flat-floor electric packaging enables wide dual modular spring-loaded holders without encroaching on driver legroom."
            },
            {
                "type": "CONFLICT",
                "title": "Tesla Supercharger (NACS) Standardization vs Proprietary BMW Connected Charging",
                "description": "Integrating native NACS port requires 800V-to-400V booster hardware at older V3 Superchargers, increasing powertrain BOM.",
                "resolution_strategy": "Standard factory NACS inlet with integrated high-voltage boost converter and ISO 15118-20 automated Plug & Charge."
            },
            {
                "type": "UNCERTAINTY",
                "title": "Federal EV Tax Credit and Regulatory Policy Shifts",
                "description": "Changes in Clean Vehicle credits impact US buyer monthly lease calculations.",
                "resolution_strategy": "Ensure BMW Financial Services commercial lease pass-through qualification and Spartanburg battery cell localization."
            }
        ],
        "Global": [
            {
                "type": "CONFLICT",
                "title": "Global Platform Commonality vs Hyper-Localized Regional Derivative Costs",
                "description": "Developing distinct regional variants (China LWB G68 with rear fridge vs Europe high-speed acoustic package vs US NACS port) risks ballooning development cost and factory complexity.",
                "resolution_strategy": "Modular Neue Klasse zonal compute architecture where hardware derivatives share common mounting and software API layers."
            },
            {
                "type": "UNCERTAINTY",
                "title": "Global High-Power DC Charging Network Reliability & Winter Preconditioning",
                "description": "Public DC fast-charging uptime varies drastically between Germany, US, and China.",
                "resolution_strategy": "Automated route-based battery preconditioning and Plug & Charge authentication across all regional roaming networks."
            }
        ]
    }
    
    # Candidate Features by Region
    candidate_features_map = {
        "China": [
            {
                "name": "Integrated Rear Console Refrigerator (8.5L Mini-Fridge)",
                "cluster": "382 customer complaints on warm beverages during executive trips in G68. Direct parity with Li Auto L8 dual-compressor fridge."
            },
            {
                "name": "DeepSeek / Alibaba Local LLM Voice Engine (≤300ms Latency)",
                "cluster": "514 complaints regarding 2-3s Mandarin voice latency in BMW OS 8.5/9. Grounded in Annual Report p. 29 localized software commitment."
            },
            {
                "name": "Amap Lane-Level AR Guidance on Panoramic Vision HUD",
                "cluster": "Navigation verbatim cluster: Complex multi-tier elevated highway guidance in Shanghai/Beijing flyovers."
            },
            {
                "name": "Rapid Pull-Down Active Seat Climate & Acupressure System",
                "cluster": "Summer cabin heat soak verbatims: Seat surface cooling from 38°C to comfortable within ≤90 seconds."
            }
        ],
        "Europe": [
            {
                "name": "Dual-Laminated Acoustic Double Glazing & Active Aero Shielding",
                "cluster": "340 Autobahn wind flutter complaints above 130 km/h. Standardizing SA 3KA acoustic package to beat Mercedes EQE benchmark."
            },
            {
                "name": "Tactile Haptic Quick-Access Control Bar & Steering Micro-Controllers",
                "cluster": "Euro NCAP 2026 physical button safety protocol & 462 verbatims rejecting screen submenu HVAC hunting at 160 km/h."
            },
            {
                "name": "Neue Klasse 800V High-Power Charging (10-80% in ≤15 min)",
                "cluster": "Winter charging curve throttling verbatims: Automated high-power preconditioning on Autobahn navigation waypoints."
            },
            {
                "name": "High-Efficiency Winter Heat Pump with Cold-Weather Range Shield",
                "cluster": "Real-world winter range drop verbatims: Gen-6 cylindrical cells limiting freezing range loss to ≤18%."
            }
        ],
        "USA": [
            {
                "name": "Native NACS Charge Inlet with ISO 15118-20 Plug & Charge",
                "cluster": "420 verbatims on broken non-Tesla DC chargers. Instant access to 15,000+ Tesla Superchargers in <5 seconds plug-to-charge."
            },
            {
                "name": "Configurable Center Storage with 36oz Insulated Tumbler Clearance",
                "cluster": "310 verbatims on large travel mugs blocking wireless phone charging tray. Dual 98mm cup clearance with cooled Qi2 cradle."
            },
            {
                "name": "Hands-Free Highway Assistant (UN Reg 171 Eye-Activated Lane Changes)",
                "cluster": "Interstate cruising feedback on I-95/I-80: Smooth hands-free cruising up to 130 km/h (Annual Report p. 54)."
            },
            {
                "name": "Neue Klasse 800V High-Power Charging (Adds 200+ Miles in 10 Minutes)",
                "cluster": "Long-distance road trip anxiety: Rapid charging replenishment matching gas stop duration."
            }
        ],
        "Global": [
            {
                "name": "Neue Klasse 800V High-Power Charging (10-80% in ≤15 min)",
                "cluster": "Universal 800V silicon-carbide platform and Gen-6 cylindrical cells delivering 720 km WLTP / 420 miles EPA range."
            },
            {
                "name": "Dual-Laminated Acoustic Double Glazing & Active Aero Road-Noise Shielding",
                "cluster": "Universal interior acoustic sound pressure ceiling ≤61 dB(A) at 130 km/h."
            },
            {
                "name": "Rapid Pull-Down Active Seat Climate & Acupressure System",
                "cluster": "High-efficiency dual blower seating reducing occupant contact surface temperature by ≥10°C in ≤90 seconds."
            },
            {
                "name": "Thermally-Cooled Qi2 Magnetic Wireless Phone Charging Cradle",
                "cluster": "Eliminates smartphone overheating and CarPlay battery throttling across all global smartphone sizes."
            }
        ]
    }
    
    # Competitors
    comps_for_region = []
    if target_region in ["China", "Europe", "USA"]:
        comps_for_region = series_comps.get(target_region, [])
    else:
        for r_name in ["China", "Europe", "USA"]:
            comps_for_region.extend(series_comps.get(r_name, [])[:1])
            
    # Regional Weights Profile
    weights_profile = req_engine.REGIONAL_WEIGHTS.get(target_region, req_engine.REGIONAL_WEIGHTS["Global"])
    
    return {
        "metadata": data.get("metadata"),
        "region": target_region,
        "vehicle_series": "5 Series",
        "delivery_metrics": delivery_metrics_map.get(target_region, delivery_metrics_map["Global"]),
        "market_context": market_context_map.get(target_region, market_context_map["Global"]),
        "annual_report_grounding": annual_report_map.get(target_region, annual_report_map["Global"]),
        "conflicts": conflicts_map.get(target_region, conflicts_map["Global"]),
        "candidate_features": candidate_features_map.get(target_region, candidate_features_map["Global"]),
        "competitors": comps_for_region,
        "regional_weights": weights_profile,
        "annual_report": data.get("metadata", {}).get("annual_report", {})
    }


@app.get("/api/requirements")
def list_requirements(
    region: Optional[str] = Query(None, description="China | Europe | USA | Global"),
    vehicle_series: Optional[str] = Query(None, description="5 Series | 1 Series | 7 Series | All"),
    domain: Optional[str] = Query(None, description="Interior | Exterior | Powertrain | Digital_ADAS"),
    status: Optional[str] = Query(None, description="DRAFT | APPROVED | REJECTED | CHALLENGED | MODIFIED"),
    search: Optional[str] = Query(None, description="Free text keyword search")
):
    db = req_engine.load_requirements_db()
    reqs = db.get("requirements", [])
    
    # Filter by vehicle series
    if vehicle_series and vehicle_series != "All":
        reqs = [r for r in reqs if r.get("vehicle_series", "5 Series") == vehicle_series]
        
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
def export_g0_concept_brief(
    format: str = Query("json", description="json | markdown | jira_csv | jira_json"),
    region: Optional[str] = Query(None, description="China | Europe | USA | Global")
):
    import csv
    import io
    db = req_engine.load_requirements_db()
    all_reqs = db.get("requirements", [])
    
    # Filter 5 Series requirements
    reqs = [r for r in all_reqs if r.get("vehicle_series", "5 Series") == "5 Series"]
    if region and region != "Global":
        reqs = [r for r in reqs if r.get("target_region") == region or r.get("target_region") == "Global"]
        
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # 1. Jira CSV Export
    if format.lower() == "jira_csv":
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        writer.writerow([
            "Issue Type", "Issue Key", "Summary", "Description", 
            "Priority", "Component", "Status", "Labels", "Vehicle Series", "Target Region", "Priority Score"
        ])
        
        for r in reqs:
            score = r.get("prioritization_score", {})
            comp_score = score.get("composite_score", 8.0)
            priority = "Highest" if comp_score >= 9.2 else ("High" if comp_score >= 8.5 else "Medium")
            
            # Format Jira-ready description
            desc_parts = [
                f"*Vehicle:* BMW 5 Series / i5 (G60 / G68 Successor)",
                f"*Target Region:* {r.get('target_region')}",
                f"*Domain:* {r.get('domain')}",
                f"*Composite Score:* {comp_score}/10",
                "",
                "h3. SMART Specification",
                r.get("smart_specification", ""),
                "",
                "h3. Acceptance Criteria"
            ]
            for c in r.get("acceptance_criteria", []):
                desc_parts.append(f"* {c}")
                
            desc_parts.extend(["", "h3. Evidence & Grounded Citations"])
            for s in r.get("evidence", {}).get("sources", []):
                desc_parts.append(f"* [{s.get('source_type')}] {s.get('citation')}")
                
            desc_parts.extend(["", "h3. Assumptions & Uncertainties"])
            for a in r.get("assumptions_uncertainties", []):
                desc_parts.append(f"* {a}")
                
            description_text = "\n".join(desc_parts)
            labels = f"BMW-5-Series Neue-Klasse {r.get('target_region', 'Global')} {r.get('domain', 'Engineering')}"
            
            writer.writerow([
                "Story",
                r.get("requirement_id"),
                r.get("title"),
                description_text,
                priority,
                r.get("domain", "System"),
                r.get("decision_status", "DRAFT"),
                labels,
                "5 Series",
                r.get("target_region"),
                comp_score
            ])
            
        csv_content = output.getvalue()
        return {
            "format": "jira_csv",
            "filename": f"BMW_5Series_Jira_Import_{region or 'Global'}.csv",
            "content": csv_content
        }
        
    # 2. Jira REST API JSON Export
    if format.lower() == "jira_json":
        jira_issues = []
        for r in reqs:
            score = r.get("prioritization_score", {})
            comp_score = score.get("composite_score", 8.0)
            priority = "Highest" if comp_score >= 9.2 else ("High" if comp_score >= 8.5 else "Medium")
            
            jira_issues.append({
                "fields": {
                    "project": {"key": "BMW5"},
                    "summary": f"[{r.get('requirement_id')}] {r.get('title')}",
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": r.get("smart_specification", "")}]
                            }
                        ]
                    },
                    "issuetype": {"name": "Story"},
                    "priority": {"name": priority},
                    "labels": ["BMW-5Series", "Neue-Klasse", r.get("target_region", "Global"), r.get("domain")],
                    "components": [{"name": r.get("domain")}],
                    "customfield_acceptance_criteria": r.get("acceptance_criteria", []),
                    "customfield_gate_status": r.get("decision_status", "DRAFT"),
                    "customfield_priority_score": comp_score
                }
            })
        return {
            "format": "jira_json",
            "total_issues": len(jira_issues),
            "region": region or "Global",
            "issues": jira_issues
        }
    
    # 3. Markdown Gate G0/G1 Concept Brief
    if format.lower() == "markdown":
        md = []
        md.append(f"# BMW AG — Concept Freeze & Project Approval Brief (Gate G0 / G1)")
        md.append(f"**Vehicle Line:** Next-Generation BMW 5 Series / i5 (G60 / G68 Successor)")
        md.append(f"**Regional Scope:** {region or 'Global Market Harmonization'}")
        md.append(f"**Platform Architecture:** BMW Neue Klasse (800V Architecture)")
        md.append(f"**Generated Date:** {timestamp}")
        md.append(f"**Review Status:** Product Management Executive Review\n")
        md.append("---\n")
        
        md.append("## 1. Executive Summary & Market Context")
        if region == "China":
            md.append("- **Strategic Imperative:** Defend executive sedan leadership in China against domestic premium NEVs (Li Auto L8, NIO ET7, XPENG G9, Xiaomi SU7). Deliver rear cabin luxury (integrated 8.5L compressor fridge, active rapid seat climate) and localized sub-300ms Mandarin AI.")
            md.append("- **Annual Report Reference:** Grounded in BMW Group 2025 China deliveries (626k units) and localized software commitment (~70% local code with DeepSeek, Alibaba, and Amap lane-level navigation on BMW Panoramic Vision, p. 29).")
        elif region == "Europe":
            md.append("- **Strategic Imperative:** Reinforce European fleet leadership (114k+ 5 Series units). Optimize high-speed Autobahn acoustics (≤61 dB(A) at 130 km/h) and incorporate tactile haptic HVAC controls meeting Euro NCAP 2026 safety guidelines.")
            md.append("- **Annual Report Reference:** Grounded in European deliveries (+7.3% YoY to 1,017,600 units, p. 25, 79) and €8,319M R&D deployment to Neue Klasse 800V silicon-carbide efficiency.")
        elif region == "USA":
            md.append("- **Strategic Imperative:** Accelerate North American executive EV adoption. Provide native Tesla Supercharger (NACS) charging port with automated ISO 15118-20 Plug & Charge and American interior storage accommodating 36oz insulated travel tumblers.")
            md.append("- **Annual Report Reference:** Grounded in Americas delivery growth (+5.6% YoY) and Highway Assistant hands-free driving approval up to 130 km/h (p. 54).")
        else:
            md.append("- **Strategic Imperative:** Balanced global executive benchmark (299,582 units delivered in 2025, +19.5% YoY, representing 13.8% of BMW Group sales). Neue Klasse 800V architecture with market-tailored derivatives.")
            md.append("- **Annual Report Reference:** Grounded in BMW Group Annual Report 2025 (€8,319M R&D capital allocation, p. 25, 79).")
        md.append("")
        
        md.append("## 2. Derived SMART Product Requirements")
        for idx, r in enumerate(reqs, 1):
            score = r.get("prioritization_score", {})
            evidence = r.get("evidence", {})
            md.append(f"### {idx}. [{r.get('requirement_id')}] {r.get('title')}")
            md.append(f"- **Region:** {r.get('target_region')} | **Domain:** {r.get('domain')} | **Status:** **{r.get('decision_status')}**")
            md.append(f"- **Composite Priority Score:** {score.get('composite_score')}/10 (Demand: {score.get('customer_demand')}, Urgency: {score.get('competitive_urgency')}, Brand: {score.get('brand_alignment')}, Effort: {score.get('effort_proxy')})")
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
        return {"format": "markdown", "filename": f"BMW_5Series_Gate_G0_Brief_{region or 'Global'}.md", "content": md_text}
        
    return {
        "brief_metadata": {
            "title": f"BMW 5 Series (G60/G68) Gate G0/G1 Concept Brief — {region or 'Global'}",
            "date": timestamp,
            "vehicle_line": "5 Series / i5",
            "platform": "Neue Klasse 800V",
            "target_region": region or "Global",
            "total_requirements": len(reqs),
            "approved_count": len([r for r in reqs if r.get("decision_status") == "APPROVED"])
        },
        "requirements": reqs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
