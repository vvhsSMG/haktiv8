import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Routing Bot (Chat)", layout="wide")
load_dotenv()

OSRM_BASE_DEFAULT = "https://router.project-osrm.org"
JAKARTA_TZ = timezone(timedelta(hours=7))
MAX_WAYPOINTS_PER_GMAPS_URL = 20
DEFAULT_MAP_THEME = "Light"

if "chat" not in st.session_state:
    st.session_state.chat = []
if "plan" not in st.session_state:
    st.session_state.plan = {
        "start": None,
        "end": None,
        "loop_back": False,
        "stops": [],
        "mode": "driving",
        "optimize": False,
        "depart_at": None,
        "osrm_base": OSRM_BASE_DEFAULT,
        "map_theme": DEFAULT_MAP_THEME,
    }
if "result" not in st.session_state:
    st.session_state.result = None

def _fmt_coord_osrm(lat, lon):
    return f"{lon:.6f},{lat:.6f}"

@st.cache_data(show_spinner=False)
def osrm_trip(base, coords, profile, roundtrip, source, destination):
    url = f"{base}/trip/v1/{profile}/" + ";".join(_fmt_coord_osrm(*c) for c in coords)
    params = {"roundtrip": "true" if roundtrip else "false", "source": source, "destination": destination, "overview": "false"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("trips"):
        raise RuntimeError(json.dumps(data))
    waypoints = data["waypoints"]
    pairs = sorted([(i, wp.get("waypoint_index", i)) for i, wp in enumerate(waypoints)], key=lambda x: x[1])
    order = [i for i, _ in pairs]
    return {"order": order, "trip": data["trips"][0]}

@st.cache_data(show_spinner=False)
def osrm_route(base, coords, profile):
    url = f"{base}/route/v1/{profile}/" + ";".join(_fmt_coord_osrm(*c) for c in coords)
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(json.dumps(data))
    return data["routes"][0]

def build_schedule(legs, places_df, depart_at_iso):
    t0 = datetime.fromisoformat(depart_at_iso).astimezone(JAKARTA_TZ) if depart_at_iso else datetime.now(JAKARTA_TZ)
    rows = [{"name": places_df.iloc[0]["name"], "role": "start", "arrive": t0, "depart": t0, "leg_distance_km": 0.0, "leg_drive_min": 0.0, "stay_min": 0}]
    for i, leg in enumerate(legs, start=0):
        drive_sec = float(leg["duration"])
        dist_km = float(leg["distance"]) / 1000.0
        arrival = rows[-1]["depart"] + timedelta(seconds=drive_sec)
        nxt = places_df.iloc[i + 1]
        stay_min = int(nxt.get("stay_min") or 0) if nxt["role"] == "stop" else 0
        depart = arrival + timedelta(minutes=stay_min)
        rows.append({"name": nxt["name"], "role": nxt["role"], "arrive": arrival, "depart": depart, "leg_distance_km": dist_km, "leg_drive_min": drive_sec / 60.0, "stay_min": stay_min})
    sched = pd.DataFrame(rows)
    return sched, float(sched["leg_distance_km"].sum()), float(sched["leg_drive_min"].sum())

def gmaps_url_from_coords(coords, mode="driving"):
    def latlon(c): return f"{c[0]:.6f},{c[1]:.6f}"
    origin = latlon(coords[0])
    destination = latlon(coords[-1])
    waypoints = "|".join(latlon(c) for c in coords[1:-1]) if len(coords) > 2 else ""
    base = "https://www.google.com/maps/dir/?api=1"
    parts = [f"origin={origin}", f"destination={destination}", f"travelmode={mode}"]
    if waypoints:
        parts.append(f"waypoints={waypoints}")
    return base + "&" + "&".join(parts)

def chunk_coords_for_gmaps(coords, max_wp_between=MAX_WAYPOINTS_PER_GMAPS_URL):
    if len(coords) <= max_wp_between + 2:
        return [coords]
    segs = []
    i = 0
    while i < len(coords) - 1:
        j = min(i + max_wp_between + 1, len(coords) - 1)
        segs.append(coords[i:j + 1])
        i = j
    return segs

def summarize_plan(plan):
    lines = []
    lines.append(f"Mode: {plan['mode']} | Optimize: {plan['optimize']}")
    lines.append(f"Depart at: {plan['depart_at'] or 'now (WIB)'}")
    if plan["start"]:
        s = plan["start"]; lines.append(f"Start: {s['name']} ({s['lat']:.6f},{s['lon']:.6f})")
    if plan["loop_back"]:
        lines.append("End: same as Start (loop)")
    elif plan["end"]:
        e = plan["end"]; lines.append(f"End: {e['name']} ({e['lat']:.6f},{e['lon']:.6f})")
    if plan["stops"]:
        lines.append("Stops:")
        for i, p in enumerate(plan["stops"], 1):
            lines.append(f"  {i}) {p['name']} | ({p['lat']:.6f},{p['lon']:.6f}) | {p['stay_min']} min")
    else:
        lines.append("Stops: (none)")
    return "\n".join(lines)

def _add_or_update_stop_in_plan(name, lat, lon, stay_min):
    stops = st.session_state.plan["stops"]
    key = (round(float(lat), 6), round(float(lon), 6), name.strip().lower())
    for s in stops:
        if (round(float(s["lat"]), 6), round(float(s["lon"]), 6), s["name"].strip().lower()) == key:
            s["stay_min"] = int(stay_min)
            return "updated"
    stops.append({"name": name, "lat": float(lat), "lon": float(lon), "stay_min": int(stay_min)})
    return "added"

def compute_route(plan, optimize):
    places = []
    if not plan["start"]:
        raise ValueError("Start is not set.")
    start = plan["start"]; end = plan.get("end")
    if end and abs(start["lat"] - end["lat"]) < 1e-6 and abs(start["lon"] - end["lon"]) < 1e-6:
        plan["loop_back"] = True
        plan["end"] = None
        end = None
    places.append({"name": start["name"], "lat": start["lat"], "lon": start["lon"], "stay_min": 0, "role": "start"})
    for p in plan["stops"]:
        places.append({"name": p["name"], "lat": p["lat"], "lon": p["lon"], "stay_min": int(p.get("stay_min") or 0), "role": "stop"})
    loop_back = bool(plan["loop_back"]) or (plan.get("end") is None)
    if not loop_back and end:
        places.append({"name": end["name"], "lat": end["lat"], "lon": end["lon"], "stay_min": 0, "role": "end"})
    else:
        places.append({"name": "Start (End)", "lat": start["lat"], "lon": start["lon"], "stay_min": 0, "role": "end"})
    coords = [(p["lat"], p["lon"]) for p in places]
    if optimize:
        trip = osrm_trip(st.session_state.plan["osrm_base"], coords, plan["mode"], roundtrip=loop_back, source="first", destination="last")
        order = trip["order"]
        ordered_places = [places[i] for i in order]
        ordered_coords = [(p["lat"], p["lon"]) for p in ordered_places]
        if loop_back and ordered_coords[-1] != ordered_coords[0]:
            ordered_coords.append(ordered_coords[0])
            ordered_places.append({"name": "Back to Start", "lat": ordered_places[0]["lat"], "lon": ordered_places[0]["lon"], "stay_min": 0, "role": "end"})
    else:
        ordered_places = places[:]
        ordered_coords = coords[:]
        if loop_back and ordered_coords[-1] != ordered_coords[0]:
            ordered_coords.append(ordered_coords[0])
            ordered_places.append({"name": "Back to Start", "lat": ordered_places[0]["lat"], "lon": ordered_places[0]["lon"], "stay_min": 0, "role": "end"})
    route = osrm_route(st.session_state.plan["osrm_base"], ordered_coords, profile=plan["mode"])
    legs = route["legs"]
    sched_input = pd.DataFrame(ordered_places)
    schedule_df, total_km, total_drive_min = build_schedule(legs, sched_input, plan["depart_at"])
    coords_for_links = ordered_coords
    if len(coords_for_links) >= 2 and coords_for_links[-1] == coords_for_links[0]:
        coords_for_links = coords_for_links[:-1]
    segments = chunk_coords_for_gmaps(coords_for_links, MAX_WAYPOINTS_PER_GMAPS_URL)
    links = [gmaps_url_from_coords(seg, plan["mode"]) for seg in segments]
    return {"ordered_places": ordered_places, "ordered_coords": ordered_coords, "route": route, "schedule_df": schedule_df, "total_km": total_km, "total_drive_min": total_drive_min, "links": links}

def _make_map_deck(path_lonlat, ordered_places, theme):
    if theme == "Dark":
        tile_url = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        route_color = [255, 215, 0]
        stop_fill_default = [0, 200, 255]
        start_fill = [0, 255, 0]
        end_fill = [0, 0, 0]
        text_color = [255, 255, 255]
    else:
        tile_url = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        route_color = [31, 119, 180]
        stop_fill_default = [220, 20, 60]
        start_fill = [0, 128, 0]
        end_fill = [0, 0, 0]
        text_color = [0, 0, 0]

    tile_layer = pdk.Layer("TileLayer", data=tile_url)
    route_layer = pdk.Layer("PathLayer", data=[{"path": path_lonlat}], get_path="path", get_color=route_color, width_scale=1, width_min_pixels=5, pickable=False)

    pts = []
    labels = []
    for idx, p in enumerate(ordered_places):
        color = stop_fill_default
        if p["role"] == "start":
            color = start_fill
        if p["role"] == "end":
            color = end_fill
        pts.append({"pos": [p["lon"], p["lat"]], "name": p["name"], "color": color})
        labels.append({"position": [p["lon"], p["lat"]], "text": f"{idx}", "color": text_color})

    stops_layer = pdk.Layer("ScatterplotLayer", data=pts, get_position="pos", get_fill_color="color", get_line_color=[0, 0, 0], line_width_min_pixels=1, get_radius=35, radius_min_pixels=6, pickable=True)
    text_layer = pdk.Layer("TextLayer", data=labels, get_position="position", get_text="text", get_color="color", get_size=16, size_units="pixels", get_text_anchor="'middle'", get_alignment_baseline="'center'", get_pixel_offset=[0, 0])

    vlat, vlon = ordered_places[0]["lat"], ordered_places[0]["lon"]
    view_state = pdk.ViewState(latitude=vlat, longitude=vlon, zoom=11)
    return pdk.Deck(layers=[tile_layer, route_layer, stops_layer, text_layer], initial_view_state=view_state, map_style=None, tooltip={"text": "{name}"})

@tool
def set_start(lat: float, lon: float, name: str = "Start") -> str:
    """Set the start point using latitude, longitude, and optional name."""
    st.session_state.plan["start"] = {"name": name, "lat": float(lat), "lon": float(lon)}
    return f"Start set to {name} ({lat:.6f},{lon:.6f})."

@tool
def set_end(lat: float | None = None, lon: float | None = None, name: str = "End", loop: bool | None = None) -> str:
    """Set the end point with latitude and longitude, or loop=True to return to start."""
    if loop is True or (lat is None and lon is None):
        st.session_state.plan["end"] = None
        st.session_state.plan["loop_back"] = True
        return "End set to loop back to Start."
    if lat is None or lon is None:
        return "Provide both lat and lon for end, or loop=True."
    st.session_state.plan["end"] = {"name": name, "lat": float(lat), "lon": float(lon)}
    st.session_state.plan["loop_back"] = False
    return f"End set to {name} ({lat:.6f},{lon:.6f})."

@tool
def add_stop(name: str, lat: float, lon: float, stay_min: int = 30) -> str:
    """Add or update a stop with name, latitude, longitude, and stay time in minutes."""
    r = _add_or_update_stop_in_plan(name, lat, lon, stay_min)
    return f"{'Updated' if r=='updated' else 'Added'} stop: {name} | ({lat:.6f},{lon:.6f}) | {stay_min} min."

@tool
def remove_stop(name: str | None = None, index: int | None = None) -> str:
    """Remove a stop by name or by 1-based index."""
    stops = st.session_state.plan["stops"]
    if index is not None:
        i = int(index) - 1
        if 0 <= i < len(stops):
            s = stops.pop(i)
            return f"Removed stop #{index}: {s['name']}."
        return "Index out of range."
    if name:
        for i, s in enumerate(stops):
            if s["name"].lower() == name.lower():
                stops.pop(i)
                return f"Removed stop: {name}."
        return f"Stop not found: {name}."
    return "Provide a stop name or index to remove."

@tool
def clear_stops() -> str:
    """Clear all stops from the plan."""
    st.session_state.plan["stops"] = []
    return "All stops cleared."

@tool
def set_stops_bulk(bulk: str, replace: bool = False) -> str:
    """Add multiple stops in format (Name|lat, lon|stay_min), optionally replacing existing."""
    text = (bulk or "").strip()
    if replace:
        st.session_state.plan["stops"] = []
    if not text:
        return "No stops provided."
    entries = re.findall(r"\(([^()]+)\)", text)
    if not entries:
        parts = re.split(r"\n|;", text)
        entries = [p.strip().strip(",") for p in parts if "|" in p]
    added, updated = 0, 0
    for e in entries:
        e = re.sub(r"\s*\|\s*", "|", e.strip().strip(","))
        fields = [f.strip() for f in e.split("|")]
        if len(fields) < 3:
            continue
        name = fields[0].strip().strip("'").strip('"')
        latlon = fields[1]
        try:
            lat_str, lon_str = [x.strip() for x in latlon.split(",", 1)]
            lat = float(lat_str); lon = float(lon_str)
            stay = int(re.sub(r"[^\d\-]", "", fields[2]))
        except Exception:
            continue
        res = _add_or_update_stop_in_plan(name, lat, lon, stay)
        if res == "added":
            added += 1
        else:
            updated += 1
    return f"Bulk processed: {added} added, {updated} updated."

@tool
def set_mode(mode: str) -> str:
    """Set travel mode to driving, walking, or cycling."""
    m = mode.strip().lower()
    if m not in {"driving", "walking", "cycling"}:
        return "Mode must be: driving, walking, cycling."
    st.session_state.plan["mode"] = m
    return f"Mode set to {m}."

@tool
def set_optimize(optimize: bool) -> str:
    """Enable or disable stop order optimization."""
    st.session_state.plan["optimize"] = bool(optimize)
    return f"Optimize set to {bool(optimize)}."

@tool
def set_depart_at(value: str) -> str:
    """Set departure time in ISO 8601; accepts time-only like 10:00 or 10:00 AM; WIB assumed if timezone is missing."""
    s = (value or "").strip()
    if not s:
        return "Invalid datetime."
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JAKARTA_TZ)
        st.session_state.plan["depart_at"] = dt.isoformat()
        return f"Departure set to {st.session_state.plan['depart_at']}."
    except Exception:
        pass
    try:
        x = s.upper().replace(".", "")
        if x.endswith("AM") or x.endswith("PM"):
            if not (" " in x[-3:] or ":" in x[-5:]):
                x = x[:-2] + " " + x[-2:]
        parsed = None
        for fmt in ("%H:%M", "%H%M", "%I:%M %p", "%I %p", "%I:%M%p", "%I%p"):
            try:
                parsed = datetime.strptime(x, fmt)
                break
            except Exception:
                continue
        if not parsed:
            return "Invalid time."
        now = datetime.now(JAKARTA_TZ)
        candidate = datetime(now.year, now.month, now.day, parsed.hour, parsed.minute, tzinfo=JAKARTA_TZ)
        if candidate < now - timedelta(minutes=5):
            candidate = candidate + timedelta(days=1)
        st.session_state.plan["depart_at"] = candidate.isoformat()
        return f"Departure set to {st.session_state.plan['depart_at']}."
    except Exception:
        return "Invalid time."

@tool
def set_osrm_base(url: str) -> str:
    """Set the OSRM base URL."""
    st.session_state.plan["osrm_base"] = url.strip() or OSRM_BASE_DEFAULT
    return f"OSRM set to {st.session_state.plan['osrm_base']}."

@tool
def set_map_theme(theme: str) -> str:
    """Set map theme to Light or Dark."""
    t = theme.strip().title()
    if t not in {"Light", "Dark"}:
        return "Theme must be Light or Dark."
    st.session_state.plan["map_theme"] = t
    return f"Theme set to {t}."

@tool
def get_next_question() -> str:
    """Return the next required question in the planning sequence as JSON."""
    plan = st.session_state.plan
    if not plan.get("start"):
        return json.dumps({"stage": "start", "ready": False, "question": "Provide START: lat, lon (and optional name)."})
    if plan.get("end") is None and not plan.get("loop_back", False):
        return json.dumps({"stage": "end_or_loop", "ready": False, "question": "LOOP back to start? If yes say 'loop'. Otherwise provide END: lat, lon (and optional name)."})
    if len(plan.get("stops", [])) == 0:
        return json.dumps({"stage": "stops", "ready": False, "question": "Add STOPS (name | lat, lon | stay_min). Or say 'no stops'."})
    if plan.get("mode") not in {"driving", "walking", "cycling"}:
        return json.dumps({"stage": "mode", "ready": False, "question": "Choose MODE: driving / walking / cycling."})
    if not plan.get("depart_at"):
        return json.dumps({"stage": "depart_at", "ready": False, "question": "Provide DEPART time (ISO 8601 or time-only like 10:00 / 10:00 AM)."})
    summary = summarize_plan(plan)
    return json.dumps({"stage": "confirm", "ready": True, "question": "Confirm to compute (say 'compute' or 'yes'), or edit.\n\n" + summary})

@tool
def show_plan() -> str:
    """Show the current plan summary."""
    return summarize_plan(st.session_state.plan)

@tool
def compute() -> str:
    """Compute the route with the current plan settings."""
    plan = st.session_state.plan
    if len(plan.get("stops", [])) == 0:
        return "No stops found. Please add at least one stop (name | lat, lon | stay_min) before computing."
    result = compute_route(plan, plan.get("optimize", False))
    st.session_state.result = result
    order_names = " -> ".join(p["name"] for p in result["ordered_places"])
    return "ROUTE_READY\n" + f"Order: {order_names}\n" + f"Total: {result['total_km']:.1f} km, {result['total_drive_min']:.0f} min\n" + "\n".join(f"GMaps {i+1}: {u}" for i, u in enumerate(result["links"]))

@tool
def optimize_and_recompute() -> str:
    """Enable optimization and recompute the route."""
    plan = st.session_state.plan
    plan["optimize"] = True
    result = compute_route(plan, True)
    st.session_state.result = result
    order_names = " -> ".join(p["name"] for p in result["ordered_places"])
    return "OPTIMIZED_ROUTE_READY\n" + f"Order: {order_names}\n" + f"Total: {result['total_km']:.1f} km, {result['total_drive_min']:.0f} min\n" + "\n".join(f"GMaps {i+1}: {u}" for i, u in enumerate(result["links"]))

TOOLS = [
    set_start, set_end, add_stop, remove_stop, clear_stops, set_stops_bulk,
    set_mode, set_optimize, set_depart_at, set_osrm_base, set_map_theme,
    get_next_question, show_plan, compute, optimize_and_recompute,
]

def _get_gemini_key_via_ui():
    seed = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if "api_key" not in st.session_state:
        st.session_state.api_key = seed
    with st.sidebar:
        st.subheader("API Key")
        key_in = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
        if st.button("Use key"):
            st.session_state.api_key = (key_in or "").strip()
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
    return st.session_state.api_key

GOOGLE_API_KEY = _get_gemini_key_via_ui()
if not GOOGLE_API_KEY:
    st.info("Enter your Gemini API key in the sidebar.")
    st.stop()

SYSTEM_PROMPT = (
    "You are RoutingBot. You help plan routes using latitude/longitude.\n"
    "Follow this sequence: 1) START, 2) END or LOOP, 3) STOPS, 4) MODE, 5) DEPART TIME, 6) CONFIRM, 7) COMPUTE.\n"
    "After each user message, call get_next_question to decide what to ask or which tool to call.\n"
    "When stage='confirm' and ready=true and user confirms, call compute.\n"
    "If user says 'optimize', call optimize_and_recompute.\n"
    "If the user pastes '(Name|lat, lon|stay)', call set_stops_bulk (replace=true only if they explicitly say replace/overwrite).\n"
    "Never say you cannot show a map. The UI renders the itinerary and map automatically after compute.\n"
    "Keep replies concise."
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=GOOGLE_API_KEY, temperature=0.2)
llm_with_tools = llm.bind_tools(TOOLS)

def _plan_ready(plan: dict) -> bool:
    return bool(
        plan.get("start") and
        (plan.get("loop_back") or plan.get("end") is not None) and
        len(plan.get("stops", [])) > 0 and
        plan.get("mode") in {"driving", "walking", "cycling"} and
        plan.get("depart_at")
    )



def _prehandle_user_text(user_text: str) -> str | None:
    txt = user_text.strip()
    logs = []

    m = re.search(r"(?i)\bstart\s*:\s*([-+.\d]+)\s*,\s*([-+.\d]+)(?:\s*\|\s*(.+))?", txt)
    if m:
        lat = float(m.group(1)); lon = float(m.group(2)); name = (m.group(3) or "Start").strip()
        logs.append(str(set_start.invoke({"lat": lat, "lon": lon, "name": name})))

    if re.search(r"(?i)^\s*loop\s*$", txt) or re.search(r"(?i)\bend\s*:\s*$", txt):
        logs.append(str(set_end.invoke({"loop": True})))

    m = re.search(r"(?i)\bend\s*:\s*([-+.\d]+)\s*,\s*([-+.\d]+)(?:\s*\|\s*(.+))?", txt)
    if m:
        lat = float(m.group(1)); lon = float(m.group(2)); name = (m.group(3) or "End").strip()
        logs.append(str(set_end.invoke({"lat": lat, "lon": lon, "name": name})))

    # Single "Add stop: ..." line
    m = re.search(r"(?i)\badd\s+stop\s*:\s*(.+)", txt)
    if m:
        logs.append(str(set_stops_bulk.invoke({"bulk": m.group(1), "replace": False})))

    # Bulk stops
    has_stop_list = bool(re.search(r"\([^()]*\|[^()]*\|[^()]*\)", txt)) or ("|" in txt and "," in txt and "(" not in txt)
    if has_stop_list:
        replace = bool(re.search(r"(?i)\breplace\b|\boverwrite\b", txt)) and not re.search(r"(?i)\bno\b", txt)
        logs.append(str(set_stops_bulk.invoke({"bulk": txt, "replace": replace})))

    # Mode (also accept bare "driving"/"walking"/"cycling")
    m = re.search(r"(?i)\bmode\s*:\s*(driving|walking|cycling)", txt)
    if m:
        logs.append(str(set_mode.invoke({"mode": m.group(1)})))
    else:
        m2 = re.fullmatch(r"\s*(driving|walking|cycling)\s*", txt, flags=re.I)
        if m2:
            logs.append(str(set_mode.invoke({"mode": m2.group(1)})))

    # Depart (also accept time-only message like "10:00 AM")
    m = re.search(r"(?i)\bdepart(?:\s*time)?\s*:\s*(.+)", txt)
    if m:
        logs.append(str(set_depart_at.invoke({"value": m.group(1).strip()})))
    else:
        if re.fullmatch(r"\s*\d{1,2}(:\d{2})?\s*(?:AM|PM|am|pm)?\s*", txt):
            logs.append(str(set_depart_at.invoke({"value": txt.strip()})))

    # Theme / OSRM
    m = re.search(r"(?i)\btheme\s*:\s*(light|dark)", txt)
    if m:
        logs.append(str(set_map_theme.invoke({"theme": m.group(1)})))
    m = re.search(r"(?i)\bosrm\s*:\s*(https?://\S+)", txt)
    if m:
        logs.append(str(set_osrm_base.invoke({"url": m.group(1)})))

    # Decide Compute vs Optimize on "yes"
    lower_txt = txt.lower().strip()
    is_yes = lower_txt in {"yes", "y", "ok", "okay", "sure", "please", "yes please", "go ahead"}
    last_assistant = ""
    for m in reversed(st.session_state.chat):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break
    last_assistant_l = last_assistant.lower()

    if re.search(r"(?i)\boptimi[sz]e\b", txt):
        logs.append(str(optimize_and_recompute.invoke({})))
    elif re.search(r"(?i)\bcompute\b", txt) or re.search(r"(?i)\bconfirm\b.*\b(yes|y)\b", txt):
        logs.append(str(compute.invoke({})))
    elif is_yes:
        if "optimize" in last_assistant_l and "recompute" in last_assistant_l:
            logs.append(str(optimize_and_recompute.invoke({})))
        elif _plan_ready(st.session_state.plan):
            logs.append(str(compute.invoke({})))

    return "\n".join(logs) if logs else None



def run_agent_turn(user_text: str) -> str:
    preload = _prehandle_user_text(user_text)
    history = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in st.session_state.chat:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    if preload:
        history.append(AIMessage(content=preload))
    history.append(HumanMessage(content=user_text))
    messages = history
    for _ in range(8):
        ai = llm_with_tools.invoke(messages)
        if getattr(ai, "tool_calls", None):
            messages.append(ai)
            for call in ai.tool_calls:
                name = call["name"]; args = call.get("args", {})
                tool_f = {t.name: t for t in TOOLS}.get(name)
                if not tool_f:
                    messages.append(ToolMessage(tool_call_id=call["id"], content=f"Unknown tool: {name}", name=name))
                    continue
                try:
                    result = tool_f.invoke(args)
                except Exception as e:
                    result = f"ERROR: {e}"
                messages.append(ToolMessage(tool_call_id=call["id"], content=str(result), name=name))
            continue
        messages.append(ai)
        return ai.content
    return "I couldn't complete the step sequence. Please try again."

st.title("🧭 Routing Bot — Chat")

with st.sidebar:
    st.subheader("Settings")
    st.session_state.plan["osrm_base"] = st.text_input("OSRM Base", value=st.session_state.plan["osrm_base"])
    theme_options = ["Light", "Dark"]
    theme_idx = theme_options.index(st.session_state.plan["map_theme"]) if st.session_state.plan["map_theme"] in theme_options else 0
    st.session_state.plan["map_theme"] = st.selectbox("Map Theme", theme_options, index=theme_idx)

with st.sidebar.expander("📘 Cheat Sheet", expanded=False):
    st.markdown(
        """
**Set start / end**
- `Start: -6.175392, 106.827153`
- `Loop`
- `End: -6.200000, 106.820000 | Office`

**Add stops**
- `Add stop: Name | -6.146358, 106.845811 | 60`
- Bulk:
(Jakarta International Expo|-6.146358, 106.845811|120),
(Kota Tua|-6.135074, 106.813667|70),
(Grand Indonesia Mall|-6.196685, 106.822477|60)
- Overwrite: `Replace existing stops: yes`

**Travel mode**
- `Mode: driving`

**Departure time**
- `Depart: 2026-08-18T10:00:00+07:00`
- `Depart: 10:00 AM`

**Compute & optimize**
- `Compute`
- `Optimize`

**Edits**
- `Make RS Premier Jatinegara first`
- `Remove stop: Grand Indonesia Mall`
- `Clear stops`

**Map & engine**
- `Theme: Dark`
- `OSRM: https://router.project-osrm.org`
      """
  )

for m in st.session_state.chat:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

u = st.chat_input("Tell me your start/end and stops. Example: Start: -6.175392,106.827153 | loop | (Expo|-6.14636,106.84581|120)")
if u:
    st.session_state.chat.append({"role": "user", "content": u})
    with st.chat_message("user"):
        st.markdown(u)
    try:
        reply = run_agent_turn(u)
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"
    st.session_state.chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

res = st.session_state.result
if res:
    st.subheader("Itinerary")
    df = res["schedule_df"].copy()
    df["arrive"] = df["arrive"].dt.strftime("%Y-%m-%d %H:%M")
    df["depart"] = df["depart"].dt.strftime("%Y-%m-%d %H:%M")
    df.rename(columns={"name": "Place", "role": "Role", "leg_distance_km": "Leg km", "leg_drive_min": "Drive min", "stay_min": "Stay min"}, inplace=True)
    st.dataframe(df, use_container_width=True)
    st.markdown(f"**Total:** {res['total_km']:.1f} km • {res['total_drive_min']:.0f} min")
    st.subheader("Route Preview")
    deck = _make_map_deck(res["route"]["geometry"]["coordinates"], res["ordered_places"], st.session_state.plan["map_theme"])
    st.pydeck_chart(deck, use_container_width=True)
    st.subheader("Open in Google Maps")
    for i, url in enumerate(res["links"], 1):
        st.markdown(f"[Google maps URL {i}]({url})")

