🧭 Routing Bot (Chat)

A Streamlit chatbot that plans routes between geographic coordinates using OSRM. It can optionally optimize stop order, builds a time-aware itinerary, renders the route on a pydeck map (Light/Dark themes), and generates Google Maps navigation links. Powered by a Gemini LLM that guides users step-by-step or accepts free-form inputs.

✨ Features

Conversational planner
Collects start/end (or loop-back), stops with stay times, travel mode, and departure time.

OSRM integration

/route → path geometry + leg distances & times

/trip → stop order optimization (keeps first/last fixed)

Schedule builder
Combines drive time and stay duration into arrival/departure timestamps.

Map preview (pydeck)
Light/Dark basemaps, centered numeric labels for each stop.

Google Maps deep links
One or more links (auto-chunked to respect waypoint limits).

Robust input parsing
Accepts free-form messages and bulk stop lists; time-only inputs assume WIB (Jakarta).

Safety nets
If the LLM “forgets” to call tools, a fallback computes the route when the plan is ready.

🧩 How it works
1) Chat + pre-parser

Your messages are pre-parsed to immediately apply:
start/end/loop, stops, mode, depart time, theme, OSRM base.
This keeps the plan in sync before the LLM responds.

2) LLM tools

The chatbot uses tools (functions) to manage the plan and run routing:

Category	Tools
Plan points	set_start, set_end (loop supported)
Stops	add_stop, remove_stop, clear_stops, set_stops_bulk
Settings	set_mode, set_depart_at, set_optimize, set_osrm_base, set_map_theme
Flow	get_next_question, show_plan
Routing	compute, optimize_and_recompute
3) Routing & optimization

Build ordered places: start → stops → end/loop.

If optimization is on, call OSRM /trip (first/last fixed).

Call OSRM /route for geometry + leg metrics.

4) UI render

Itinerary table (arrival/departure per stop).

pydeck map (path + labeled markers 0,1,2…).

Google Maps links (chunked when needed).
