# BMW PM Intelligence & Requirements Co-Pilot

AI-powered product decision-making platform for BMW Product Managers, built for the **EHL Hackathon Munich (BMW Product Management Challenge)**.

## Scope & Target Vehicle
- **Vehicle Line:** Next-Generation BMW 5 Series / i5 (G60 / G68 successor, 3–5 year horizon).
- **Target Markets:** Primary focus on **China (CN)** against domestic premium NEVs (NIO, XPENG, Li Auto, Xiaomi), with extensibility to **Europe (EU)** and **USA (US)**.
- **Platform Architecture:** BMW *Neue Klasse* (800V fast-charging, Gen-6 cylindrical cell packaging, BMW Panoramic Vision, and central computing architecture).

## Core Capabilities
1. **Multi-Source Ingestion:** Ingests 14,000+ verbatims (`G60`, `G70`, `F70`), customer studies, sales forecasts, and BMW 2025 Annual Report strategic facts.
2. **SMART Requirements Engine:** Translates customer complaints and competitor deltas into quantified, acceptance-tested engineering targets across 4 vehicle domains (Interior, Exterior, Powertrain, Digital & ADAS).
3. **Regional Prioritization:** Dynamic multi-factor scoring with regional weighting profiles (China vs. Europe vs. USA vs. Global).
4. **Honest AI & Uncertainty Flags:** Explicitly surfaces conflicting regional signals and evidence gaps.
5. **PM-in-the-Loop & Immutable Audit Trail:** Allows PMs to Approve, Reject, Challenge, or Modify requirements with mandatory rationale logging and complete traceability.

## Repository Structure
```
├── backend/            # FastAPI backend, data parsers & prioritization engine
├── frontend/           # Modern executive PM cockpit UI
├── BMW_Data/           # Provided hackathon datasets (verbatims, studies, option lists)
├── BMW-Group-Report-2025-en.pdf # BMW Group Annual Report 2025
├── requirements.txt    # Python dependencies
└── .gitignore          # Git exclusion rules
```
