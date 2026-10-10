"""Curated customer-need taxonomy and source mappings (human-maintained data).

Derived from the actual G60 data: the 221 VFC level-2 names in
``G60_feedback_hackathon.xlsx``, the 15 public observation IDs, the 18
competitor-evidence rows and the survey attribute strings.

* VFC names are mapped to one need category. Where the VFC name is ambiguous
  or a broad catch-all, the mapping is ``review_required`` with a reason;
  nothing is forced into an unrelated category.
* Cross-source links (public feedback, competitor evidence, survey attributes)
  are curated per need with an evidence tier and a review flag. Themes, VFC
  names and survey attributes are NOT assumed to mean the same thing.

Version strings are stored with every row so later changes are traceable.
"""
from __future__ import annotations

from dataclasses import dataclass

TAXONOMY_VERSION = "2026.10-v1"
MAPPING_VERSION = "2026.10-v1"
MISSING_VFC = "<missing>"  # key for rows whose VFC level-2 cell is empty


@dataclass(frozen=True)
class NeedCategory:
    code: str
    name: str
    description: str


TAXONOMY: list[NeedCategory] = [
    NeedCategory("INFOTAINMENT", "Infotainment & Digital Experience",
                 "Displays and their content, navigation, audio, voice assistant, smartphone projection, in-car apps."),
    NeedCategory("CONTROLS", "Physical Controls & Usability",
                 "How functions are operated: operating concept, touch/physical controls, steering-wheel buttons, selector."),
    NeedCategory("DRIVING_DYNAMICS", "Driving Dynamics & Ride",
                 "Handling, steering, suspension, ride comfort, braking feel, drive modes."),
    NeedCategory("POWERTRAIN", "Powertrain & Performance",
                 "Engine/e-motor performance, transmission, start-stop, engine sound, fuel system."),
    NeedCategory("CHARGING", "Charging Experience",
                 "Home and public charging, cables, charging socket, charging-optimised routing."),
    NeedCategory("RANGE_EFFICIENCY", "Electric Range & Efficiency",
                 "Electric range, high-voltage battery, preconditioning, recuperation, efficiency."),
    NeedCategory("SOFTWARE", "Software Reliability & Updates",
                 "Software updates, map updates, and malfunctions of software-driven functions. "
                 "Mostly identified by AI extraction because VFC level 2 rarely names software directly."),
    NeedCategory("DRIVER_ASSISTANCE", "Driver Assistance & Parking",
                 "Assisted driving (ACC, lane, highway assistant), warnings, parking assistance and cameras."),
    NeedCategory("SAFETY_SECURITY", "Safety & Security",
                 "Perceived safety, restraint systems, child safety, emergency call, anti-theft."),
    NeedCategory("CONNECTIVITY", "Connectivity & Connected Services",
                 "My BMW App, remote services, phone pairing, Wi-Fi/eSIM, USB and phone charging, ConnectedDrive."),
    NeedCategory("ACCESS", "Vehicle Access & Keys",
                 "Keys, Digital Key, Comfort Access, locking, doors, tailgate opening."),
    NeedCategory("CLIMATE", "Climate & Thermal Comfort",
                 "Air conditioning, blower, seat/steering-wheel heating and ventilation, auxiliary heating."),
    NeedCategory("INTERIOR_COMFORT", "Seating & Interior Comfort",
                 "Seats and adjustment, massage, interior comfort, cabin acoustics (NVH), sun protection."),
    NeedCategory("INTERIOR_QUALITY", "Interior Design & Quality",
                 "Interior design, materials, trim, ambient and interior lighting, perceived quality."),
    NeedCategory("PRACTICALITY", "Practicality & Storage",
                 "Luggage space, storage compartments, cupholders, spaciousness, towing and utility."),
    NeedCategory("EXTERIOR", "Exterior Design & Lighting",
                 "Exterior design and body parts, exterior lighting, mirrors, glass and wipers."),
    NeedCategory("OVERALL", "Overall Vehicle Impression",
                 "Broad statements about the whole vehicle or BMW-level catch-all VFC classes."),
    NeedCategory("OTHER", "Other / Unclassified",
                 "No usable classification (BMW 'no_class_found' or empty VFC cell)."),
]
NEED_CODES = {c.code for c in TAXONOMY}
NEED_NAMES = {c.code: c.name for c in TAXONOMY}

# ---------------------------------------------------------------------------
# VFC level 2 -> need category.  Entries in _REVIEW carry a reason.
# ---------------------------------------------------------------------------

_DIRECT: dict[str, list[str]] = {
    "INFOTAINMENT": [
        "Android Auto", "Apple CarPlay", "Apps (in-car)", "Audio mode, sound", "CD / DVD changer, playback",
        "ConnectedDrive and Infotainment, overall technology", "Intelligent Personal Assistant",
        "Intelligent Personal Assistant, activation word", "Navigation", "Radio function", "Rear seat entertainment",
        "Screen mirroring", "Smart POI search", "Video: playback / streaming", "Voice control / speech recognition",
        "Real Time Traffic Information (RTTI)", "Traffic Camera Information",
    ],
    "CONTROLS": [
        "Controller operation", "Gesture control", "Light switch operating unit", "Multi-function steering wheel operation",
        "Operate selector lever", "Operating concept, operating system", "Pedals, operate", "Shift paddle",
        "Touch screen, operation", "Wash / wipe switch", "Roof function centre",
    ],
    "DRIVING_DYNAMICS": [
        "Braking", "Cushioning / damping vehicle", "Drive, driving dynamics and chassis, overall technology",
        "Handling / Riding", "Self-leveling suspension", "Steering",
        "ground\xa0clearance",  # exact source spelling (non-breaking space)
    ],
    "POWERTRAIN": [
        "Drivetrain", "Transmission", "Engine acoustics / engine sound", "Automatic engine start-stop function (MSA)",
        "Launch Control", "Starting engine", "Start engine/switch on drive readiness", "Exhaust system",
    ],
    "CHARGING": [
        "Bidirectional charging", "Charge high-voltage battery", "Charging cable", "High-voltage charging power socket",
        "Opening/closing charging socket cover", "Public charging / external providers", "Route optimized for charging",
        "Wallbox, charging / home charging",
    ],
    "RANGE_EFFICIENCY": [
        "Electric Range", "High voltage battery", "High voltage battery, preconditioning",
        "Brake Energy Regeneration / recuperation",
    ],
    "SOFTWARE": ["Remote Software Upgrade/RSU", "Map update"],
    "DRIVER_ASSISTANCE": [
        "(Active) cruise control", "All-round view", "Back-up Assistant", "Blind spot monitoring",
        "Cross Traffic Warning Rear with Brake Intervention", "Driver assistance, automated driving, overall technology",
        "Fatigue Alert", "Front collision warning", "Highway assistant", "Intersection Warning with Brake Intervention",
        "Lane Change Assistant", "Lane departure warning", "Night Vision", "Park Distance Control (PDC)",
        "Parking Assistant", "Rear view camera", "Rear-end collision warning", "Remote controlled parking", "Side View",
        "Speed Limit Assistant", "Steering and lane guide assist", "Surround View", "Emergency Stop Assistant",
    ],
    "SAFETY_SECURITY": [
        "Anti-theft alarm system", "Child restraint seat, use", "Child safety latch", "Emergency request",
        "Impression of safety", "Seat belts", "Vehicle safety, overall technology", "Tire pressure control",
    ],
    "CONNECTIVITY": [
        "BMW ConnectedDrive upgrades/ MINI CONNECTED upgrades", "BMW Online / MINI Online",
        "ConnectedDrive services (smartphone) technology overall", "Customer Account (My BMW / MINI Portal)",
        "Manage / pair mobile device (Bluetooth)", "My BMW / MINI Portal", "My BMW App / MINI App", "Personal eSIM",
        "Priorities for telephones", "Remote 3D View (My BMW App / MINI App)",
        "Remote Engine Start (My BMW App / MINI App)", "Remote Services & Remote Keyless Entry (My BMW App/MINI App)",
        "Remote interior camera (My BMW App / MINI App)", "Teleservices", "Wi-Fi connection", "WiFi hotspot",
        "Apps (smartphone)",
    ],
    "ACCESS": [
        "Comfort Access", "Digital Key", "Display key", "Vehicle key", "Vehicle, lock / unlock", "Door, open / close",
        "Automatic doors", "non-contact trunk lid opening", "Trunk lid, open / close",
    ],
    "CLIMATE": [
        "Air conditioning, overall technology", "Air conditioning, setting", "Blower function", "Vehicle, climatization",
        "Seating, ventilate / heat", "Heated steering wheel", "Auxiliary heater function", "Panel heating",
        "Heated rear window",
    ],
    "INTERIOR_COMFORT": [
        "Seats", "Seat massage", "Seat memory", "Adjustment of lower lumbar support", "Headrest, adjust",
        "Interior comfort", "Interior acoustics", "Sun protection", "Sun visor, front",
    ],
    "INTERIOR_QUALITY": [
        "Interior design", "Upholstery / interior trim", "Ambient light", "Vehicle interior, lighting",
        "Door, finisher panel, front", "Floor mat",
    ],
    "PRACTICALITY": [
        "Luggage compartment", "Storage compartments", "Cupholder", "Glove box", "Frunk", "Space / spatial impression",
        "Through-loading facility", "12 V connection", "Trunk light", "Towing eye",
    ],
    "EXTERIOR": [
        "Exterior design", "Exterior lighting", "Fog light", "Flashing", "Exterior rearview mirror", "Bumper, front",
        "Bumper, rear", "Radiator grille", "Hood", "Roof",
    ],
    "OTHER": ["no_class_found", MISSING_VFC],
}

_REVIEW: dict[str, tuple[str, str]] = {
    # name: (need_code, reason)
    "Head-Up Display": ("INFOTAINMENT", "Display content vs. operation/readability (CONTROLS)."),
    "Instrument cluster": ("INFOTAINMENT", "Display content vs. readability/operation (CONTROLS)."),
    "Display, time": ("INFOTAINMENT", "Generic display class."),
    "Driver profile": ("INFOTAINMENT", "Personalisation settings; could be CONNECTIVITY or SOFTWARE."),
    "Owner's Manual": ("INFOTAINMENT", "Documentation (digital/print); weak fit to any category."),
    "Driver's Guide App": ("INFOTAINMENT", "Digital owner's manual; could be CONNECTIVITY."),
    "Traffic Light And Sign Warning": ("DRIVER_ASSISTANCE", "Information feature; could be INFOTAINMENT."),
    "Touch Command": ("CONTROLS", "Rear-seat touch control unit; also INFOTAINMENT."),
    "Air conditioning control panel": ("CONTROLS", "Operation of climate controls; also CLIMATE (cf. public P001)."),
    "Steering wheel": ("CONTROLS", "Physical object: could be controls, comfort or interior quality."),
    "Dynamic Stability Control": ("DRIVING_DYNAMICS", "Chassis control system with safety function."),
    "Wheels / tires": ("DRIVING_DYNAMICS", "Wheel design (EXTERIOR) vs. tyre behaviour (DRIVING_DYNAMICS)."),
    "RSC tire": ("DRIVING_DYNAMICS", "Run-flat tyre: ride comfort vs. safety/practicality."),
    "My Modes / driving experience": ("DRIVING_DYNAMICS", "Modes combine drive settings with ambience/display."),
    "Experience Modes": ("DRIVING_DYNAMICS", "Modes combine drive settings with ambience/display."),
    "Automatic-hold": ("DRIVING_DYNAMICS", "Brake-hold function; could be DRIVER_ASSISTANCE."),
    "Parking brake": ("DRIVING_DYNAMICS", "Brake operation; could be CONTROLS."),
    "Engine oil": ("POWERTRAIN", "Maintenance topic."),
    "Fuel tank": ("POWERTRAIN", "Fuel system / range for ICE."),
    "Fuel filler flap , open / close": ("POWERTRAIN", "Body part operation; could be ACCESS/EXTERIOR."),
    "Vehicle battery": ("POWERTRAIN", "12 V battery / electrical system; discharge issues may be SOFTWARE."),
    "Cooling": ("CLIMATE", "Ambiguous: cabin cooling vs. powertrain cooling."),
    "Drive Recorder": ("DRIVER_ASSISTANCE", "Camera recording feature; could be SAFETY_SECURITY."),
    "Interior camera": ("SAFETY_SECURITY", "Monitoring/security camera; could be CONNECTIVITY."),
    "Mobile device, wireless charging": ("CONNECTIVITY", "Phone charging pad; could be PRACTICALITY."),
    "USB port": ("CONNECTIVITY", "Phone connection/charging; could be PRACTICALITY."),
    "Vehicle, access and exit": ("ACCESS", "Ingress/egress ergonomics (INTERIOR_COMFORT) vs. access systems."),
    "Remote Engine Start (Vehicle Key )": ("ACCESS", "Key function for pre-conditioning; also CLIMATE/CONNECTIVITY."),
    "Door, front": ("ACCESS", "Door as a part: operation (ACCESS) vs. body/design or quality."),
    "Hood, open / close": ("ACCESS", "Body part operation; could be EXTERIOR."),
    "Resting state after door opening": ("ACCESS", "Vehicle wake/sleep behaviour; may be SOFTWARE."),
    "Garage door opener": ("ACCESS", "Convenience function; could be CONNECTIVITY."),
    "Interior mirror, garage door opener": ("ACCESS", "Convenience function in mirror; could be CONNECTIVITY."),
    "Outer door handle, front": ("ACCESS", "Access function vs. exterior design (cf. US survey 'Exterior door handles')."),
    "Outer door handle, rear": ("ACCESS", "Access function vs. exterior design."),
    "Side window, open / close": ("ACCESS", "Window operation; could be CONTROLS."),
    "Caring Car": ("INTERIOR_COMFORT", "Wellness programme combining climate, seats, ambience."),
    "Center armrest, front": ("INTERIOR_COMFORT", "Comfort vs. storage (PRACTICALITY)."),
    "Grab handle": ("INTERIOR_COMFORT", "Small interior part."),
    "Panorama glass roof": ("INTERIOR_COMFORT", "Ambience/light (comfort) vs. design."),
    "Interior": ("INTERIOR_QUALITY", "Broad interior catch-all class."),
    "Instrument panel": ("INTERIOR_QUALITY", "Dashboard design/quality vs. displays."),
    "Center console, front": ("INTERIOR_QUALITY", "Design/quality vs. storage or controls."),
    "Interior rearview mirror": ("INTERIOR_QUALITY", "Small part; could be EXTERIOR/visibility."),
    "Side sill trim panel": ("INTERIOR_QUALITY", "Trim part; interior vs. exterior unclear."),
    "Trunk lid": ("PRACTICALITY", "Trunk lid as part vs. its operation (ACCESS)."),
    "Tirefit": ("PRACTICALITY", "Tyre repair kit; could be SAFETY_SECURITY."),
    "Trailer, use": ("PRACTICALITY", "Towing utility."),
    "Running board": ("PRACTICALITY", "Unusual for a sedan; may be mis-coded."),
    "Opening / closing convertible top / hardtop": ("PRACTICALITY", "Not applicable to a G60 sedan; likely mis-coded."),
    "Exterior": ("EXTERIOR", "Broad exterior catch-all class."),
    "Windshield": ("EXTERIOR", "Glass part; could be visibility/quality."),
    "Rear wiper": ("EXTERIOR", "Wiper function; G60 sedan has no rear wiper - may be mis-coded."),
    "Windshield wiper": ("EXTERIOR", "Wiper function; could be CONTROLS."),
    "Rain sensor mode": ("EXTERIOR", "Automatic wiper function; could be CONTROLS."),
    "Washer fluid, top up": ("EXTERIOR", "Maintenance topic."),
    "Horns": ("EXTERIOR", "Warning device; could be SAFETY_SECURITY."),
    "Welcome Light Carpet": ("EXTERIOR", "Exterior lighting effect on approach; also ACCESS."),
    "High-beam assistant": ("DRIVER_ASSISTANCE", "Assistance function affecting EXTERIOR lighting (cf. public P009)."),
    "Body equipment, overall technology": ("OVERALL", "BMW catch-all class covering many systems; not a single need."),
    "Accessories / Parts": ("OVERALL", "Catch-all."),
    "General enquiry: customer feedback": ("OVERALL", "Catch-all."),
}


def vfc_mapping() -> dict[str, tuple[str, str, str | None]]:
    """name -> (need_code, mapping_status, rationale)."""
    mapping: dict[str, tuple[str, str, str | None]] = {}
    for code, names in _DIRECT.items():
        for name in names:
            if name in mapping:
                raise ValueError(f"VFC name mapped twice: {name}")
            reason = "No usable VFC classification." if code == "OTHER" else None
            mapping[name] = (code, "direct", reason)
    for name, (code, reason) in _REVIEW.items():
        if name in mapping:
            raise ValueError(f"VFC name mapped twice: {name}")
        mapping[name] = (code, "review_required", reason)
    bad = {c for c, _, _ in mapping.values()} - NEED_CODES
    if bad:
        raise ValueError(f"Unknown need codes in mapping: {bad}")
    return mapping


# ---------------------------------------------------------------------------
# Cross-source curated links: need -> evidence. (key, review_required, note)
# ---------------------------------------------------------------------------

# Public owner feedback, by original public ID (BMW_5_Series_Public_Owner_Feedback.xlsx).
PUBLIC_FEEDBACK_LINKS: dict[str, list[tuple[str, bool, str | None]]] = {
    "CONTROLS": [("P001", False, "Missing physical climate buttons."), ("P003", False, None),
                 ("P006", True, "Distraction from manual infotainment interactions."), ("P008", False, None)],
    "CLIMATE": [("P001", True, "Theme is climate *controls* - operation, not climate performance.")],
    "INFOTAINMENT": [("P003", False, None), ("P006", False, None), ("P008", False, "OS 8.5 menu navigation."),
                     ("P010", False, "Audio system quality."), ("P015", False, "Voice control.")],
    "PRACTICALITY": [("P002", False, None)],
    "INTERIOR_COMFORT": [("P004", False, "Seat comfort part of the observation."),
                         ("P011", True, "Cabin quietness part of the observation."), ("P012", False, "Wind noise (NVH)."),
                         ("P014", False, None)],
    "POWERTRAIN": [("P004", True, "Acceleration part of the observation."), ("P007", False, None)],
    "RANGE_EFFICIENCY": [("P004", True, "Fuel economy (ICE), not electric range."),
                         ("P007", True, "Self-reported efficiency (ICE)."), ("P011", True, "Fuel consumption (diesel).")],
    "OVERALL": [("P005", True, "Perceived feature regression vs. older 5 Series.")],
    "DRIVER_ASSISTANCE": [("P009", False, "Automatic high-beam behaviour.")],
    "EXTERIOR": [("P009", True, "Lighting behaviour, not lighting design.")],
    "CHARGING": [("P013", False, "Noise during fast charging (i5).")],
}

# Competitor-evidence rows, by worksheet row in BMW_G60_Public_Competitor_Evidence.xlsx.
# BMW rows (2-6) are public BMW owner evidence -> SUPPORTING; others -> CONTEXTUAL.
COMPETITOR_LINKS: dict[str, list[tuple[int, bool, str | None]]] = {
    "INFOTAINMENT": [(2, False, None), (7, False, None), (10, False, None), (12, True, "Mixed cabin/technology summary."),
                     (13, True, "Platform topic list, not issue frequency.")],
    "CONTROLS": [(2, False, "Nested menus, touch-operated vents."), (5, False, None), (7, False, "Small capacitive controls.")],
    "INTERIOR_QUALITY": [(3, False, None), (18, True, "Glass crack; quality vs. exterior.")],
    "DRIVING_DYNAMICS": [(4, False, "Ride/suspension comfort."), (6, False, None), (13, True, "Driving experience topic.")],
    "RANGE_EFFICIENCY": [(4, True, "Consumption of a 520d (ICE).")],
    "INTERIOR_COMFORT": [(6, True, "Seats and ride."), (11, False, None), (12, True, "Quiet ride."), (14, False, None)],
    "PRACTICALITY": [(8, False, "Rear legroom."), (11, False, "Trunk space.")],
    "DRIVER_ASSISTANCE": [(8, False, "Unexpected adaptive cruise braking."), (19, False, None)],
    "SOFTWARE": [(9, True, "Electronics warnings; root cause unknown."), (17, True, "Low-voltage failure; electrical.")],
    "POWERTRAIN": [(14, False, None), (15, True, "Alleged drivetrain failure; unverified complaint.")],
    "SAFETY_SECURITY": [(15, True, "Safety-system alerts; unverified complaint.")],
    "EXTERIOR": [(16, False, "Headlamp failure."), (18, True, "Sunroof glass crack.")],
}

# Survey attributes (study_name, attribute exactly as stored). Survey items are
# perception ratings - related to, not identical with, VFC feedback classes.
STUDY_LINKS: dict[str, list[tuple[str, str, bool]]] = {
    "INFOTAINMENT": [
        ("CN_EU_2025", "Features of information and entertainment system including in-car apps", False),
        ("CN_EU_2025", "Easy to use information and entertainment system including in-car apps", True),
        ("CN_EU_2025", "Integrated navigation system", False),
        ("US_2025", "Overall performance of sound system", False), ("US_2025", "Audio clarity", False),
        ("US_2025", "If applicable: Navigation system", False),
        ("US_2025", "Mirroring of smartphone content on vehicle display", False),
        ("US_2025", "Head-up Display", False), ("US_2025", "Position/ integration of displays in dashboard", True),
        ("US_2025", "Menu structure of infotainment system", False),
        ("US_2025", "If applicable: Interactive information system (e.g., OnStar, mbrace, iDrive)", False),
        ("US_2025", "Performance of voice recognition", False),
    ],
    "CONTROLS": [
        ("CN_EU_2025", "Easy to read and use instruments/ controls/ displays", False),
        ("CN_EU_2025", "Easy to use information and entertainment system including in-car apps", True),
        ("US_2025", "Placement/ ease of reading and using controls", False), ("US_2025", "Intuitiveness of controls", False),
        ("US_2025", "Overall driver-vehicle interaction (HMI)", False),
        ("US_2025", "Ease of reading and using screen interface", False),
        ("US_2025", "Operation of heater/ AC controls", False),
        ("US_2025", "Menu structure of infotainment system", True),
    ],
    "DRIVING_DYNAMICS": [
        ("CN_EU_2025", "Roadholding", False), ("CN_EU_2025", "Braking", False), ("CN_EU_2025", "Manoeuvrability", False),
        ("CN_EU_2025", "Quality of ride", False), ("CN_EU_2025", "Sportiness", True),
        ("US_2025", "Overall driving performance", False), ("US_2025", "Riding comfort", False),
        ("US_2025", "Turning radius", False), ("US_2025", "Handling/Maneuverability", False), ("US_2025", "Braking", False),
        ("US_2025", "Ground clearance", False),
    ],
    "POWERTRAIN": [
        ("CN_EU_2025", "Acceleration", False), ("US_2025", "Overall power and pickup", False),
        ("US_2025", "Acceleration from stop", False), ("US_2025", "Passing capability", False),
        ("US_2025", "Smoothness of transmission", False), ("US_2025", "Sound of engine", True),
    ],
    "RANGE_EFFICIENCY": [
        ("CN_EU_2025", "Fuel economy/ energy efficiency", True),
        ("US_2025", "Gas or electric mileage (fuel economy)", True),
    ],
    "SOFTWARE": [
        ("CN_EU_2025", "Reliability", True), ("US_2025", "Freedom from electrical problems", True),
        ("US_2025", "Overall impression of durability/reliability", True),
    ],
    "DRIVER_ASSISTANCE": [
        ("US_2025", "Overall driver assist features", False),
        ("US_2025", "If applicable: Backup system (camera, sensors, etc.)", False),
        ("US_2025", "Safety monitoring features (blid spot warning, lane departure warning, etc)", False),
        ("US_2025", "Adaptive driving assists (lane keeping, cruise, braking)", False),
    ],
    "SAFETY_SECURITY": [("CN_EU_2025", "Safety", False), ("US_2025", "Overall safety of the vehicle", False)],
    "CONNECTIVITY": [
        ("CN_EU_2025", "Connected services", False), ("CN_EU_2025", "Smartphone integration/ connectivity", False),
        ("US_2025", "Bluetooth/ smartphone pairing", False), ("US_2025", "WiFi hotspot in vehicle", False),
        ("US_2025", "Vehicle's Mobile App", False), ("US_2025", "Phone charging", False),
    ],
    "ACCESS": [
        ("CN_EU_2025", "Ease of entry into and out of car", True), ("US_2025", "Exterior door handles", True),
        ("US_2025", "Trunk lid/ tailgate and liftgate operations", True),
    ],
    "CLIMATE": [
        ("CN_EU_2025", "Heating/ ventilation/ air conditioning ", False),
        ("US_2025", "Overall performance of climate control system", False), ("US_2025", "Heater performance", False),
        ("US_2025", "Air conditioner performance", False),
    ],
    "INTERIOR_COMFORT": [
        ("CN_EU_2025", "Overall seating comfort", False), ("CN_EU_2025", "Quietness when driving", False),
        ("CN_EU_2025", "Driving position", False), ("US_2025", "Overall comfort of the seats", False),
        ("US_2025", "Comfort of front seat", False), ("US_2025", "Comfort of 2nd row seat", False),
        ("US_2025", "Driver seat adjustability", False), ("US_2025", "Overall noise/vibration/harshness", False),
        ("US_2025", "Freedom from wind and road noise", False), ("US_2025", "Provides a sense of wellbeing", True),
    ],
    "INTERIOR_QUALITY": [
        ("CN_EU_2025", "Interior styling", False), ("CN_EU_2025", "Quality of interior and materials used", False),
        ("CN_EU_2025", "Overall build quality", True), ("US_2025", "Overall interior styling/ aesthetics", False),
        ("US_2025", "Interior trim and finish", False), ("US_2025", "Interior lighting", False),
        ("US_2025", "Overall interior workmanship", False), ("US_2025", "Quality of interior materials", False),
        ("US_2025", "Overall interior design / layout", False), ("US_2025", "Instrument panel (IP)", True),
    ],
    "PRACTICALITY": [
        ("CN_EU_2025", "Interior storage for small items", False), ("CN_EU_2025", "Rear interior roominess", False),
        ("CN_EU_2025", "Ease of loading/ unloading luggage", False), ("CN_EU_2025", "Luggage capacity", False),
        ("CN_EU_2025", "Interior versatility", False), ("US_2025", "Interior roominess", False),
        ("US_2025", "Interior storage", False), ("US_2025", "Cargo capacity/usefulness", False),
        ("US_2025", "Cupholders", False), ("US_2025", "Place for phone", False),
    ],
    "EXTERIOR": [
        ("CN_EU_2025", "Exterior styling", False), ("US_2025", "Overall exterior styling", False),
        ("US_2025", "Size/ proportions", False), ("US_2025", "Appearance of tires/ wheels", False),
        ("US_2025", "Headlight/ taillight design", False), ("US_2025", "Headlight function/ brightness", True),
    ],
    "OVERALL": [
        ("CN_EU_2025", "Overall product satisfaction", False), ("CN_EU_2025", "Level of equipment", True),
        ("CN_EU_2025", "Value for money", True), ("US_2025", "Level of standard equipment", True),
        ("US_2025", "Overall value for money", True), ("US_2025", "Technical innovations", True),
    ],
}


def sync_taxonomy(session) -> dict[str, int]:
    """Write the curated taxonomy and VFC mappings to the database (replacing
    previous versions of these curated tables only)."""
    from sqlalchemy import delete

    from core.ai_models import CustomerNeedCategory, VfcNeedMapping

    session.execute(delete(VfcNeedMapping))
    session.execute(delete(CustomerNeedCategory))
    session.flush()
    session.add_all(
        CustomerNeedCategory(need_code=c.code, name=c.name, description=c.description, sort_order=i,
                             taxonomy_version=TAXONOMY_VERSION)
        for i, c in enumerate(TAXONOMY, start=1)
    )
    session.flush()
    mapping = vfc_mapping()
    session.add_all(
        VfcNeedMapping(vfc_level2_name=name, need_code=code, mapping_status=status, rationale=reason,
                       taxonomy_version=TAXONOMY_VERSION)
        for name, (code, status, reason) in sorted(mapping.items())
    )
    session.flush()
    return {"categories": len(TAXONOMY), "vfc_mappings": len(mapping),
            "review_required": sum(1 for _, s, _ in mapping.values() if s == "review_required")}
