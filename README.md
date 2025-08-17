Routing Bot (Chat)

A Streamlit chatbot that plans routes between geographic coordinates using OSRM for routing, optionally optimizes stop order, produces a time-aware itinerary, renders the route on a pydeck map, and generates Google Maps navigation links. It’s driven by a Gemini-powered LLM that guides the user step-by-step or accepts free-form inputs.

Features

Conversational planner: Collects start/end (or loop-back), stops with stay times, travel mode, and departure time.

OSRM integration:

/route to build the path and compute leg distances/times.

/trip to optimize stop order (keeps start/end fixed).

Schedule builder: Combines drive times + stay durations to compute arrival/departure timestamps.

Map preview: Renders the route and stops in Streamlit using pydeck (Light/Dark themes, numeric labels centered).

Google Maps deep links: Creates one or more links for step-by-step navigation (handles waypoint limits).

Robust input parsing: Accepts simple sentences and bulk stop lists; time-only inputs assume Jakarta time (WIB).

Safety nets: If the LLM “forgets” to call tools, a fallback will compute the route when the plan is ready

How it works

Chat + pre-parser
Your messages are pre-parsed to immediately set start/end, stops, mode, departure, theme, and OSRM base. This ensures the plan is up to date before the LLM responds.

LLM tools
The chatbot uses a set of tools (functions) to manage the plan and run routing:

set_start, set_end (or loop)

add_stop, remove_stop, clear_stops, set_stops_bulk

set_mode, set_depart_at, set_optimize

set_osrm_base, set_map_theme

compute, optimize_and_recompute

show_plan, get_next_question

Routing + optimization

Build the ordered place list (start → stops → end/loop).

If optimization is enabled, call OSRM /trip to compute an optimal order (preserving first/last).

Call OSRM /route to get leg distances/durations + geometry for the map.

UI render

An itinerary table with arrival/departure per stop.

A pydeck map with the path and numbered markers (0,1,2…).

Google Maps links for navigation (chunked when needed).

Usage

You can chat naturally or follow a structured flow. The bot will guide you through:

Start

End or Loop

Stops (with stay minutes)

Mode (driving/walking/cycling)

Depart time (10AM or time-only)

Confirm

Compute (and optionally Optimize)

Example conversation
You: Start: -6.175392, 106.827153
You: Loop
You: (Jakarta International Expo|-6.146358, 106.845811|120),
     (Kota Tua|-6.135074, 106.813667|70),
     (Grand Indonesia Mall|-6.196685, 106.822477|60)
You: Driving
You: 10:00 AM
You: Compute


The app then shows:

A complete itinerary table (with arrival/departure times).

A map preview of your route with numeric labels.

One or more Google Maps links to navigate.
