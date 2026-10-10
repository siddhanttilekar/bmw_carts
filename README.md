# BMW PM Intelligence & Requirements Co-Pilot

AI-powered decision-support platform for BMW Product Managers, built for the **EHL Hackathon Munich (BMW Product Management Challenge)**.

---

## 1. Executive Summary & Challenge Scope
* **Target Vehicle Line:** Next-Generation BMW 5 Series / i5 (G60 / G68 Long-Wheelbase Successor, 3–5 year horizon).
* **Strategic Battleground:** Primary focus on **China (CN)** defending executive market share against domestic premium EV disruptors (NIO, Li Auto, XPENG, Xiaomi), with dynamic extensibility to **Europe (EU)** and **USA (US)**.
* **Platform Architecture:** BMW *Neue Klasse* (800V fast-charging, Gen-6 cylindrical cell packaging, BMW Panoramic Vision, and zonal central compute).

---

## 2. Alignment with BMW Automotive Product Development Process
The application directly operates at **Stage 1 (Market Research & Needs Analysis)** and **Stage 2 (Product Planning)**, preparing the formal deliverables required to pass **Gate G0 (Project Approval)** and **Gate G1 (Concept Freeze)**:

```
[ Stage 13: Market Feedback ] ──┐
  (12,000+ verbatims & studies)  │
                                 ▼
                     [ Stage 1: Situation Analysis ]
                       • 12,000+ Verbatims (G60, G70, F70)
                       • Customer Studies (CN, EU, US satisfaction)
                       • Sales Volumes & 2025 Annual Report (€8.3B R&D)
                       • Regional Competitor Deltas (NIO, Li Auto, Tesla)
                                 │
                                 ▼
                     [ Stage 2: Structure & Context ]
                       • Semantic Deduplication & Clustering
                       • Candidate Feature List (0 / 1)
                       • "Honest AI" Conflict & Uncertainty Radar
                                 │
                                 ▼
                     [ Stage 3: Requirements Definition ]
                       • Strict SMART Quantified Acceptance Metrics
                       • Domains: Interior, Exterior, Powertrain, Digital & ADAS
                                 │
                                 ▼
                     [ Stage 4: Prioritization Engine ]
                       • Multi-Factor Explainable Scoring
                       • Dynamic Regional Sensitivity Sliders
                                 │
                                 ▼
              [ Stage 5: Documentation & PM Audit Trail ]
                • PM-in-the-Loop Reviews (Approve, Reject, Challenge, Modify)
                • Immutable Historical Audit Trail
                • One-Click Gate G0 / G1 Concept Brief Export
```

---

## 3. The 4 Winning Differentiators ("Why This Solution Wins")

### 1. "Honest AI" Conflict & Uncertainty Radar 🎖️
In accordance with the hackathon evaluation criteria (*"We reward honesty. Trustworthy AI acknowledges uncertainty"*), the system explicitly flags:
* **Cultural Contradictions:** German Autobahn drivers demanding physical tactile dials vs. Chinese executives demanding pure voice/touch.
* **Engineering Trade-offs:** High-luxury active refrigerator (+3.2 kg) vs. curb weight & WLTP efficiency.
* **Sample Size Gaps:** Flags when verbatims express strong complaints but survey sample size remains low.

### 2. Multi-Market Dynamic Strategy Switcher
Instantly switches between **China 🇨🇳**, **Europe 🇪🇺**, **USA 🇺🇸**, and **Global 🌐**:
* **China Focus:** Prioritizes executive rear seating, compressor fridges, DeepSeek/Alibaba voice AI, Amap lane-level navigation, and Huawei HarmonyOS NEXT.
* **Europe Focus:** Prioritizes Autobahn high-speed acoustic insulation (≤61 dB(A) at 130 km/h), tactile quick-access controls, and WLTP energy efficiency.
* **USA Focus:** Prioritizes native NACS Tesla Supercharger access, 36oz tumbler console packaging, and hands-free highway cruising.

### 3. Strict SMART Engineering Quantifications (Zero Hallucinations)
* **Bad / Unacceptable:** *"A fridge with a lot of space"*
* **Good / Implemented:** *"Integrated rear console compressor refrigerator with 8.5L usable volume (capacity for two 0.75L wine bottles or four 330ml cans), achieving pull-down cooling from 25°C to 4°C in ≤20 minutes, heating mode up to 50°C, operating at acoustic noise ≤36 dB(A)."*

### 4. Interactive PM-in-the-Loop Sensitivity Sandbox
Live sliders for:
$$\text{Composite Score} = (W_1 \cdot \text{Customer Demand}) + (W_2 \cdot \text{Competitive Urgency}) + (W_3 \cdot \text{Brand DNA}) - (W_4 \cdot \text{Effort})$$
Allows PMs to simulate different strategic postures and watch rankings recalculate in real-time.

---

## 4. Quick Start & Execution

### Option A: One-Click Launcher (Windows)
Double-click [`start_cockpit.bat`](file:///C:/EHL%20Munich/start_cockpit.bat) or run in PowerShell:
```powershell
.\start_cockpit.bat
```
This automatically verifies dependencies, starts the FastAPI backend, and opens your default browser at `http://127.0.0.1:8000/`.

### Option B: Manual Command Line
```powershell
python backend/main.py
```
Then navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in any web browser.

---

## 5. End-to-End Verification
To verify the data processing pipeline, run:
```powershell
python backend/verify_pipeline.py
```

---

## 6. Project Structure
```
├── backend/
│   ├── data_engine.py          # Parser for verbatims, studies, sales, and annual report
│   ├── requirements_engine.py  # SMART formulation, regional weights & scoring logic
│   ├── main.py                 # FastAPI REST API & static file server
│   └── verify_pipeline.py      # Automated pipeline verification script
├── frontend/
│   └── index.html              # Modern BMW Neue Klasse executive UI
├── data/
│   ├── processed_signals.json  # Indexed customer signals & competitor specs
│   └── requirements_db.json    # SMART specifications, scores & audit logs
├── BMW_Data/                   # Hackathon input datasets
├── BMW-Group-Report-2025-en.pdf# BMW Group 2025 Annual Report
├── start_cockpit.bat           # One-click Windows launch script
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```
