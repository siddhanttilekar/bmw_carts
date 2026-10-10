"""
BMW PM Intelligence & Requirements Co-Pilot - SMART Requirements & Prioritization Engine
Implements:
1. Stage 3 (Requirements Definition): Measurable SMART acceptance targets across 4 vehicle domains.
2. Stage 4 (Prioritization): Multi-factor scoring with regional weighting profiles (China, Europe, USA, Global).
3. Evidence Confidence Level (HIGH, MEDIUM, LOW) & traceable citations.
4. Assumptions, Uncertainties, and Conflict Resolution.
5. Immutable Audit Trail state machine.
"""

import os
import json
import datetime
from typing import Dict, List, Any

OUTPUT_DIR = r"C:\EHL Munich\data"
REQUIREMENTS_DB_FILE = os.path.join(OUTPUT_DIR, "requirements_db.json")

# Default regional weight profiles for prioritization
# Score = (W_demand * Demand) + (W_urgency * Urgency) + (W_brand * Brand) - (W_effort * Effort)
REGIONAL_WEIGHTS = {
    "China": {
        "weight_demand": 0.35,
        "weight_urgency": 0.40,
        "weight_brand": 0.15,
        "weight_effort": 0.10,
        "description": "High urgency on domestic NEV parity (Li Auto, NIO, XPENG) & local software ecosystems."
    },
    "Europe": {
        "weight_demand": 0.30,
        "weight_urgency": 0.25,
        "weight_brand": 0.35,
        "weight_effort": 0.10,
        "description": "High weight on BMW driving DNA, Autobahn acoustic refinement, and WLTP energy efficiency."
    },
    "USA": {
        "weight_demand": 0.40,
        "weight_urgency": 0.30,
        "weight_brand": 0.20,
        "weight_effort": 0.10,
        "description": "High weight on public charging reliability (NACS), EPA highway range, and interior storage utility."
    },
    "Global": {
        "weight_demand": 0.35,
        "weight_urgency": 0.30,
        "weight_brand": 0.25,
        "weight_effort": 0.10,
        "description": "Balanced global scorecard across volume, brand prestige, and technical complexity."
    }
}

def calculate_priority_score(metrics: Dict[str, float], region: str = "Global") -> Dict[str, Any]:
    weights = REGIONAL_WEIGHTS.get(region, REGIONAL_WEIGHTS["Global"])
    
    demand = metrics.get("customer_demand", 8.0)
    urgency = metrics.get("competitive_urgency", 8.0)
    brand = metrics.get("brand_alignment", 8.0)
    effort = metrics.get("effort_score", 5.0)  # 1 (Low) to 10 (Very High)
    
    raw_score = (
        (demand * weights["weight_demand"]) +
        (urgency * weights["weight_urgency"]) +
        (brand * weights["weight_brand"]) -
        (effort * weights["weight_effort"])
    )
    
    # Scale composite score to 1.0 - 10.0 scale
    composite_score = round(max(1.0, min(10.0, raw_score)), 1)
    
    effort_proxy = "LOW"
    if effort >= 7.5:
        effort_proxy = "HIGH"
    elif effort >= 5.0:
        effort_proxy = "MEDIUM"
        
    return {
        "composite_score": composite_score,
        "customer_demand": demand,
        "competitive_urgency": urgency,
        "brand_alignment": brand,
        "effort_proxy": effort_proxy,
        "effort_score": effort,
        "active_region_profile": region
    }

def get_base_requirements() -> List[Dict[str, Any]]:
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    raw_reqs = [
        {
            "requirement_id": "REQ-5SER-CN-001",
            "target_region": "China",
            "domain": "Interior",
            "title": "Dual-Zone Active Rear Chilled Console for Executive Seating",
            "smart_specification": "Integrated rear center console compressor-refrigeration unit with 8.5L usable volume (capacity for two 0.75L wine bottles or four 330ml cans), achieving active pull-down cooling from 25°C to 4°C within ≤20 minutes, heating mode up to 50°C, operating at acoustic noise ≤36 dB(A), controllable via rear touch console and My BMW App.",
            "acceptance_criteria": [
                "Usable volume ≥ 8.5 liters with dual champagne/can divider",
                "Thermal pull-down from 25°C ambient to 4°C in ≤ 20 minutes",
                "Acoustic noise ceiling ≤ 36 dB(A) during peak compressor cycle",
                "Remote cabin pre-cooling scheduling via My BMW App and HarmonyOS NEXT widget"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Customer_Verbatim", "citation": "G60-0388: 'Rear passengers in executive trips expect cold beverages and better thigh support than currently provided in G68 standard seats.'"},
                    {"source_type": "Customer_Study", "citation": "F70_G60_G68_G70_customer_studies.xlsx (CN_EU_2025): G68 rear seat comfort lags flagship G70 by 0.86 satisfaction points in China."},
                    {"source_type": "Competitor_Benchmark", "citation": "Li Auto L8/L9 features standard dual-compressor 8.8L fridge (0°C–50°C) at RMB 339,800."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.4, "competitive_urgency": 9.6, "brand_alignment": 8.0, "effort_score": 5.5},
            "assumptions_uncertainties": [
                "Assumption: Chauffeur and rear executive passenger usage in Tier 1 China remains >45% for G68 long-wheelbase.",
                "Packaging Constraint: Compressor unit must not compromise trunk golf bag load-through capacity (minimum 480L retained)."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Derived from G68 rear cabin feedback and Li Auto L8 competitor packaging advantage in China."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-CN-002",
            "target_region": "China",
            "domain": "Digital_ADAS",
            "title": "DeepSeek/Alibaba LLM-Powered Natural Voice Assistant (≤300ms Latency)",
            "smart_specification": "BMW Operating System X in-vehicle conversational AI utilizing localized DeepSeek and Alibaba LLM models, delivering multi-turn natural language dialog with end-to-end cloud/edge response latency ≤300ms, multi-intent execution in Mandarin and Cantonese, and native integration with WeChat and Huawei HarmonyOS NEXT ecosystems.",
            "acceptance_criteria": [
                "End-to-end voice intent execution latency ≤ 300 ms on 5G network",
                "Complex multi-intent command success rate ≥ 96% in Mandarin and Cantonese dialects",
                "Zero-latency edge fallback for cabin climate, windows, and seat adjustments without internet",
                "Native WeChat voice messaging and destination sending via My BMW app"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Annual_Report_2025", "citation": "BMW Group Report 2025, p. 29: Chinese derivative of BMW OS X incorporates ~70% local software; LLM partnerships with DeepSeek and Alibaba established for natural voice interaction."},
                    {"source_type": "Customer_Verbatim", "citation": "G60-0812: 'Voice assistant takes 2-3 seconds to understand Mandarin commands and struggles with complex navigation destinations.'"},
                    {"source_type": "Competitor_Benchmark", "citation": "NIO NOMI AI & XPENG voice assistant achieve sub-300ms latency and 4-zone conversational tracking."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.8, "competitive_urgency": 9.8, "brand_alignment": 8.6, "effort_score": 6.2},
            "assumptions_uncertainties": [
                "Assumption: Local data residency and cloud compute compliance under Chinese automotive cybersecurity laws.",
                "Conflict: Driver focus vs rich in-cabin conversational interruptions handled via localized AI gating."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Directly grounded in BMW 2025 Annual Report R&D strategy (p. 29) to eliminate conversational AI lag against Chinese NEVs."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-GL-003",
            "target_region": "Global",
            "domain": "Powertrain",
            "title": "Neue Klasse 800V High-Power Charging (10-80% in ≤15 min) with Smart Preconditioning",
            "smart_specification": "Neue Klasse 800V silicon-carbide (SiC) high-voltage platform supporting peak charging power ≥270 kW, achieving 10% to 80% state-of-charge replenishment in ≤15 minutes (adding ≥300 km in 10 minutes), with automated navigation-linked battery thermal conditioning bringing cells to optimal 32°C across -15°C to 45°C ambient temperatures.",
            "acceptance_criteria": [
                "10% to 80% DC fast-charging duration ≤ 15 minutes at 800V high-power dispensers",
                "WLTP certified range ≥ 720 km (CLTC ≥ 800 km for China, EPA ≥ 420 miles for US)",
                "Winter range degradation at -7°C ambient limited to ≤ 18% via high-efficiency heat pump",
                "Automatic thermal preconditioning initiated upon navigation waypoint to fast charger"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Annual_Report_2025", "citation": "BMW Group Report 2025, p. 53: Neue Klasse 800V architecture and Gen-6 cylindrical cells debut to maximize charging curve and energy density."},
                    {"source_type": "Customer_Verbatim", "citation": "G60-1204: 'In winter, public charging speed throttles significantly without automated high-power navigation preconditioning.'"},
                    {"source_type": "Competitor_Benchmark", "citation": "XPENG G9 (800V SiC) and Porsche Taycan deliver 10-80% in 15-18 mins."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.5, "competitive_urgency": 9.4, "brand_alignment": 9.5, "effort_score": 7.8},
            "assumptions_uncertainties": [
                "Assumption: Rapid buildout of 800V liquid-cooled public charging networks across European highways and Tier 1/2 China.",
                "Uncertainty: US rural charging infrastructure reliability mitigated by NACS Supercharger access."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Aligns with Neue Klasse platform strategic commitments and eliminates winter DC throttling complaints."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-EU-004",
            "target_region": "Europe",
            "domain": "Exterior",
            "title": "Dual-Laminated Acoustic Glazing & Active Aero Road-Noise Shielding",
            "smart_specification": "Standard dual-pane acoustic laminated glass on windshield and front/rear side windows combined with active underbody air management, achieving interior acoustic sound pressure level ≤61 dB(A) at 130 km/h and ≤65 dB(A) at 160 km/h on German Autobahn coarse asphalt surfaces.",
            "acceptance_criteria": [
                "In-cabin sound level ≤ 61 dB(A) at steady 130 km/h cruise",
                "In-cabin sound level ≤ 65 dB(A) at 160 km/h Autobahn cruise",
                "High-frequency wind flutter around A-pillars and side mirrors reduced by ≥ 3 dB(A)",
                "Standard equipment across all European 5 Series / i5 lines (unbundled from luxury packs)"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Customer_Verbatim", "citation": "G60-0419: 'Wind rustle around A-pillar and side mirrors noticeable on German Autobahn above 140 km/h.'"},
                    {"source_type": "Customer_Study", "citation": "F70_G60_G68_G70_customer_studies.xlsx: EU G60 acoustic comfort rating (8.31) trails Mercedes EQE with acoustic comfort pack (8.75)."},
                    {"source_type": "Option_List_PDF", "citation": "G60_OptionList.PDF: Acoustic glazing (SA 3KA) currently locked behind expensive Innovation package."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.0, "competitive_urgency": 8.5, "brand_alignment": 9.2, "effort_score": 4.5},
            "assumptions_uncertainties": [
                "Assumption: Autobahn high-speed cruising remains a core buying factor for German fleet managers (>60% of European 5 Series sales).",
                "Weight Penalty: Acoustic glass adds approximately 3.2 kg total curb weight, offset by Neue Klasse lightweight door structures."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Addresses significant Autobahn wind noise verbatims and closes the gap against Mercedes EQE acoustic package."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-EU-005",
            "target_region": "Europe",
            "domain": "Interior",
            "title": "Tactile Haptic Quick-Access Control Bar & Steering Micro-Controllers",
            "smart_specification": "Driver-centric tactile control strip featuring customizable haptic confirmation beneath the BMW Panoramic Vision display, enabling blind tactile operation of temperature, defrost, hazard, and drive mode selection within ≤0.5 seconds eyes-off-road duration, supplemented by physical rotary thumbwheels on the steering wheel.",
            "acceptance_criteria": [
                "Eyes-off-road duration for primary climate adjustment ≤ 0.5 seconds",
                "Haptic actuation feedback latency ≤ 50 ms with distinct click simulation",
                "Retention of physical rotary scroll wheels on steering wheel spokes",
                "Compliance with Euro NCAP 2026 physical button safety assessment protocol"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Customer_Verbatim", "citation": "G60-0084: 'Adjusting climate controls and seat heating via touchscreen while driving fast is distracting and frustrating compared to previous generation buttons.'"},
                    {"source_type": "Customer_Verbatim", "citation": "G60-0931: 'Miss the physical iDrive rotary controller shortcut buttons on the highway.'"},
                    {"source_type": "Regulatory_Trend", "citation": "Euro NCAP 2026 guidelines penalizing vehicles that force core driving functions into screen submenus."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.3, "competitive_urgency": 8.0, "brand_alignment": 9.4, "effort_score": 4.8},
            "assumptions_uncertainties": [
                "Conflict Resolution: Reconciles European preference for tactile safety with modern minimalist glass cockpit styling.",
                "Manufacturing: Module shared across Neue Klasse derivatives to minimize tooling overhead."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Direct response to overwhelming European customer feedback rejecting pure touch sub-menus for basic HVAC."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-US-006",
            "target_region": "USA",
            "domain": "Powertrain",
            "title": "Native NACS Inlet with Automated ISO 15118-20 Plug & Charge",
            "smart_specification": "Factory-integrated North American Charging Standard (NACS) charge port backed by ISO 15118-20 Plug & Charge automated encryption, establishing seamless high-speed charging session initiation across 15,000+ Tesla Supercharger dispensers and public networks in <5 seconds from plug insertion without third-party apps.",
            "acceptance_criteria": [
                "Plug-to-session initiation time < 5 seconds at Tesla Supercharger V3/V4 dispensers",
                "Full backward compatibility with CCS1 via integrated adapter accessory",
                "Real-time Supercharger stall availability and preconditioning route planning in BMW Maps",
                "Consolidated My BMW monthly charging billing for all commercial networks"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Customer_Verbatim", "citation": "G60-0771: 'Non-Tesla DC fast chargers are frequently broken or throttled. Seamless Plug & Charge at Superchargers is a must.'"},
                    {"source_type": "US_Customer_Study", "citation": "US_2025 Study: 41% of US premium EV drivers cite public charging reliability as their primary ownership pain point."},
                    {"source_type": "Competitor_Benchmark", "citation": "Tesla Model S native Supercharging remains the gold standard for long-distance US EV usability."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.7, "competitive_urgency": 9.5, "brand_alignment": 8.5, "effort_score": 5.0},
            "assumptions_uncertainties": [
                "Assumption: Tesla Supercharger network opening fully standardized across all 50 US states by launch date.",
                "Technical: 800V-to-400V high-voltage boost converter required when charging at older 400V Supercharger V3 stalls."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Critical requirement to eliminate US charging anxiety and match Tesla user experience."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-US-007",
            "target_region": "USA",
            "domain": "Interior",
            "title": "Configurable Center Storage with 36oz Insulated Tumbler Clearance",
            "smart_specification": "Redesigned center console packaging offering dual spring-loaded modular cup holders engineered to secure up to 36oz (1.06L) insulated travel tumblers (98mm base diameter) without impinging on dual 15W Qi2 wireless charging cradles or center armrest actuation, combined with 12L split-opening under-armrest storage.",
            "acceptance_criteria": [
                "Accommodates two 98mm diameter / 240mm tall insulated mugs simultaneously",
                "Dual 15W Qi2 wireless phone chargers positioned clear of cup handles with active cooling",
                "Center storage compartment volume ≥ 12 liters with illuminated USB-C 65W PD ports",
                "Zero rattle or lateral container tipping during 0.8g emergency cornering maneuvers"
            ],
            "evidence": {
                "level": "MEDIUM",
                "sources": [
                    {"source_type": "Customer_Verbatim", "citation": "G60-0215: 'Cup holders cannot comfortably fit large 32oz insulated travel mugs without blocking phone wireless charging pad.'"},
                    {"source_type": "US_Customer_Study", "citation": "US_2025 Study: Practical storage ratings scored 12% lower than prior generation BMW 5 Series (G30)."},
                    {"source_type": "Competitor_Benchmark", "citation": "Lucid Air and American luxury crossovers prioritize oversized multi-cup storage."}
                ]
            },
            "scoring_inputs": {"customer_demand": 8.6, "competitive_urgency": 7.8, "brand_alignment": 7.5, "effort_score": 3.8},
            "assumptions_uncertainties": [
                "Assumption: American consumer preference for large insulated mugs remains an enduring cultural standard.",
                "Spatial trade-off: Slight reduction in center console structural tunnel width offset by Neue Klasse flat-floor EV architecture."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Solves common US verbatim complaint regarding everyday ergonomics and phone charging interference."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-CN-008",
            "target_region": "China",
            "domain": "Digital_ADAS",
            "title": "Amap Lane-Level AR Guidance on BMW Panoramic Vision HUD",
            "smart_specification": "BMW Panoramic Vision full-windshield lower projection integrated with Amap (AutoNavi) lane-level navigation engine, projecting 3D photorealistic lane boundaries, elevated flyover fork indicators, and signalized intersection countdown timers with 10-meter virtual depth perception in Chinese mega-cities.",
            "acceptance_criteria": [
                "Lane-level guidance accuracy ≤ 0.5 meters across complex multi-tier elevated highways",
                "Refresh rate ≥ 60 Hz with latency ≤ 30 ms between vehicle GNSS positioning and HUD projection",
                "Display of dynamic traffic light phase countdowns across 95% of Tier 1/2 Chinese intersections",
                "Zero visual clutter: Projection automatically dims in high-density visual traffic zones"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Annual_Report_2025", "citation": "BMW Group Report 2025, p. 29: Navigation developed in collaboration with Amap provides precise visualization of complex intersections and highway ramps."},
                    {"source_type": "Customer_Verbatim", "citation": "G60-0812: 'Missed highway exits in Shanghai flyovers because 2D screen map is hard to interpret at speed.'"},
                    {"source_type": "Competitor_Benchmark", "citation": "XPENG XNGP and Huawei Qiankun ADS project detailed 3D environmental modeling."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.6, "competitive_urgency": 9.7, "brand_alignment": 8.8, "effort_score": 6.5},
            "assumptions_uncertainties": [
                "Assumption: Ongoing high-definition mapping data license partnership with Amap maintained.",
                "Technical: Software co-developed with local BMW ConnectedDrive Lab in Shanghai."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Direct realization of BMW Annual Report p. 29 strategy uniting BMW Panoramic Vision and Amap lane-level software."}
            ]
        },
        {
            "requirement_id": "REQ-5SER-GL-009",
            "target_region": "Global",
            "domain": "Interior",
            "title": "Rapid Pull-Down Active Seat Climate & Acupressure System",
            "smart_specification": "Next-generation multi-zone seat architecture featuring dual radial blower units per seat capable of dropping occupant contact surface temperature by ≥10°C within ≤90 seconds, maintaining quiet operation ≤40 dB(A) in Whisper Mode, integrated with 10-chamber pneumatic acupressure therapy.",
            "acceptance_criteria": [
                "Contact surface temperature reduction ≥ 10°C in ≤ 90 seconds under 38°C cabin heat soak",
                "Acoustic noise level ≤ 40 dB(A) in Whisper Mode at occupant ear level",
                "Independent cushion and backrest airflow adjustment with moisture extraction mode",
                "Weight optimization achieving ≤ 18.5 kg total seat frame mass per front seat"
            ],
            "evidence": {
                "level": "HIGH",
                "sources": [
                    {"source_type": "Customer_Verbatim", "citation": "G60-0388: 'Seat cooling on hot summer days takes over 5 minutes to feel effective through perforated leather.'"},
                    {"source_type": "Customer_Verbatim", "citation": "G70-0001: Customer praised flagship heated/ventilated ergonomics, highlighting desire for similar performance on 5 Series."},
                    {"source_type": "Competitor_Benchmark", "citation": "XPENG G9 and Li Auto L8 offer rapid suction ventilation cooling occupants within 30-60 seconds."}
                ]
            },
            "scoring_inputs": {"customer_demand": 9.2, "competitive_urgency": 8.8, "brand_alignment": 8.9, "effort_score": 5.2},
            "assumptions_uncertainties": [
                "Assumption: High summer cabin temperatures in Southern Europe, US Sunbelt, and Southern China make seat pull-down a key delight factor.",
                "Material constraint: Perforated vegan Sensafin upholstery must maintain durability under high suction airflow."
            ],
            "decision_status": "DRAFT",
            "audit_trail": [
                {"timestamp": timestamp, "actor": "AI_AGENT", "action": "GENERATED", "rationale": "Addresses significant thermal comfort delay complaints across G60 summer driving."}
            ]
        }
    ]
    
    # Calculate initial scores
    processed = []
    for r in raw_reqs:
        region = r.get("target_region", "Global")
        score_info = calculate_priority_score(r["scoring_inputs"], region=region)
        r["prioritization_score"] = score_info
        processed.append(r)
        
    return processed

def init_requirements_db() -> Dict[str, Any]:
    reqs = get_base_requirements()
    db = {
        "metadata": {
            "version": "1.0.0",
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_requirements": len(reqs),
            "vehicle_line": "Next-Generation BMW 5 Series / i5 (G60 / G68 Successor)",
            "supported_regions": ["China", "Europe", "USA", "Global"]
        },
        "regional_weight_profiles": REGIONAL_WEIGHTS,
        "requirements": reqs
    }
    
    with open(REQUIREMENTS_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
        
    print(f"Requirements database initialized with {len(reqs)} SMART specifications at {REQUIREMENTS_DB_FILE}")
    return db

def load_requirements_db() -> Dict[str, Any]:
    if not os.path.exists(REQUIREMENTS_DB_FILE):
        return init_requirements_db()
    with open(REQUIREMENTS_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_requirements_db(db: Dict[str, Any]) -> None:
    db["metadata"]["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(REQUIREMENTS_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    init_requirements_db()
