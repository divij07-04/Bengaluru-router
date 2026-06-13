import gradio as gr
import osmnx as ox
import folium
import networkx as nx
import requests
import time
import math
import os

print("Loading Bengaluru road network...")
G = ox.load_graphml("bengaluru_graph.graphml")
print(f"Graph loaded. Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

def geocode(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address + ", Bengaluru, India",
        "format": "json",
        "limit": 1,
        "countrycodes": "in"
    }
    headers = {"User-Agent": "BengaluruRouter/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        results = response.json()
        if not results:
            return None, None
        return float(results[0]["lat"]), float(results[0]["lon"])
    except:
        return None, None

def bidirectional_astar(G, source, target):
    import heapq

    def heuristic(u, v):
        lat1, lon1 = G.nodes[u]["y"], G.nodes[u]["x"]
        lat2, lon2 = G.nodes[v]["y"], G.nodes[v]["x"]
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    open_f = [(0, source)]
    g_f = {source: 0}
    par_f = {source: None}
    closed_f = set()

    open_b = [(0, target)]
    g_b = {target: 0}
    par_b = {target: None}
    closed_b = set()

    best = float("inf")
    meeting = None

    while open_f and open_b:
        _, u = heapq.heappop(open_f)
        if u in closed_f:
            continue
        closed_f.add(u)
        if u in closed_b:
            total = g_f[u] + g_b[u]
            if total < best:
                best = total
                meeting = u
            break
        for v in G.successors(u):
            w = min(d.get("length", 1) for d in G[u][v].values())
            ng = g_f[u] + w
            if v not in g_f or ng < g_f[v]:
                g_f[v] = ng
                heapq.heappush(open_f, (ng + heuristic(v, target), v))
                par_f[v] = u

        _, u = heapq.heappop(open_b)
        if u in closed_b:
            continue
        closed_b.add(u)
        if u in closed_f:
            total = g_f[u] + g_b[u]
            if total < best:
                best = total
                meeting = u
            break
        for v in G.predecessors(u):
            w = min(d.get("length", 1) for d in G[v][u].values())
            ng = g_b[u] + w
            if v not in g_b or ng < g_b[v]:
                g_b[v] = ng
                heapq.heappush(open_b, (ng + heuristic(v, source), v))
                par_b[v] = u

    if meeting is None:
        raise ValueError("No path found.")

    path = []
    node = meeting
    while node is not None:
        path.append(node)
        node = par_f[node]
    path.reverse()
    node = par_b[meeting]
    while node is not None:
        path.append(node)
        node = par_b[node]
    return path

def find_route(start_address, end_address):
    if not start_address or not end_address:
        return "<p style='color:red'>Please enter both addresses.</p>", ""

    start_lat, start_lon = geocode(start_address)
    end_lat, end_lon = geocode(end_address)

    if not start_lat or not end_lat:
        return "<p style='color:red'>Could not find one or both addresses. Be more specific.</p>", ""

    try:
        start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        end_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)
    except Exception as e:
        return f"<p style='color:red'>Error: {str(e)}</p>", ""

    try:
        t0 = time.time()
        route_astar = nx.astar_path(G, start_node, end_node, weight="length")
        time_astar = time.time() - t0

        t0 = time.time()
        route_bidir = bidirectional_astar(G, start_node, end_node)
        time_bidir = time.time() - t0
    except Exception as e:
        return f"<p style='color:red'>Routing error: {str(e)}</p>", ""

    dist_km = sum(
        G[route_astar[i]][route_astar[i+1]][0]["length"]
        for i in range(len(route_astar)-1)
    ) / 1000
    speedup = time_astar / time_bidir if time_bidir > 0 else 0

    center = ((start_lat + end_lat) / 2, (start_lon + end_lon) / 2)
    m = folium.Map(location=center, zoom_start=14)

    folium.Marker([start_lat, start_lon], popup="Start",
                  icon=folium.Icon(color="blue", icon="home")).add_to(m)
    folium.Marker([end_lat, end_lon], popup="End",
                  icon=folium.Icon(color="red", icon="flag")).add_to(m)

    astar_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_astar]
    folium.PolyLine(astar_coords, color="blue", weight=6,
                    opacity=0.6, tooltip="Standard A*").add_to(m)

    bidir_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_bidir]
    folium.PolyLine(bidir_coords, color="red", weight=3,
                    opacity=0.9, tooltip="Bidirectional A*").add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
         background:white;padding:15px;border-radius:8px;
         border:2px solid grey;font-size:14px;color:black;">
        <b>Algorithm Comparison</b><br>
        <span style="color:blue;">&#9644;</span> Standard A*: {time_astar:.4f}s<br>
        <span style="color:red;">&#9644;</span> Bidirectional A*: {time_bidir:.4f}s<br>
        <b>Speedup: {speedup:.2f}x faster</b><br>
        Distance: {dist_km:.2f} km
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_html = m._repr_html_()

    stats_html = f"""
    <div style="background:#1a1a2e;color:white;padding:20px;border-radius:10px;font-family:monospace;">
        <h3 style="color:#00d4ff;margin-top:0;">Route Found</h3>
        <p>Distance: <b>{dist_km:.2f} km</b></p>
        <p>Standard A*: <b>{time_astar:.4f}s</b> ({len(route_astar)} nodes)</p>
        <p>Bidirectional A*: <b>{time_bidir:.4f}s</b></p>
        <p style="color:#00ff88;font-size:1.2em;">Speedup: <b>{speedup:.2f}x faster</b></p>
    </div>
    """

    return stats_html, map_html

with gr.Blocks(
    title="Bengaluru Route Finder",
    css=".gradio-container {background: #0f0f1a} label {color: white !important}"
) as app:
    gr.Markdown("# Bengaluru Optimal Route Finder")
    gr.Markdown("Bidirectional A* vs Standard A* on real OpenStreetMap data (154,902 nodes)")

    with gr.Row():
        start_input = gr.Textbox(
            label="Start Address",
            placeholder="e.g. Indiranagar, Bengaluru"
        )
        end_input = gr.Textbox(
            label="End Address",
            placeholder="e.g. Electronic City, Bengaluru"
        )

    find_btn = gr.Button("Find Optimal Route", variant="primary", size="lg")
    stats_output = gr.HTML()
    map_output = gr.HTML()

    find_btn.click(
        fn=find_route,
        inputs=[start_input, end_input],
        outputs=[stats_output, map_output]
    )

app.launch()