# venue_graph.py
import networkx as nx
import numpy as np
import json

EXIT_NODES = [19, 20, 21, 22, 23, 24]

def create_stadium_graph():
    G = nx.DiGraph()

    nodes = [
        (0,  "Main Stage",       "zone",     5000, (5, 5)),
        (1,  "North Stand A",    "section",  800,  (5, 9)),
        (2,  "North Stand B",    "section",  800,  (3, 9)),
        (3,  "North Stand C",    "section",  800,  (7, 9)),
        (4,  "South Stand A",    "section",  800,  (5, 1)),
        (5,  "South Stand B",    "section",  800,  (3, 1)),
        (6,  "South Stand C",    "section",  800,  (7, 1)),
        (7,  "East Stand",       "section",  1200, (9, 5)),
        (8,  "West Stand",       "section",  1200, (1, 5)),
        (9,  "North Concourse",  "corridor", 400,  (5, 8)),
        (10, "South Concourse",  "corridor", 400,  (5, 2)),
        (11, "East Concourse",   "corridor", 400,  (8, 5)),
        (12, "West Concourse",   "corridor", 400,  (2, 5)),
        (13, "North Tunnel L",   "tunnel",   150,  (3, 7)),
        (14, "North Tunnel R",   "tunnel",   150,  (7, 7)),
        (15, "South Tunnel L",   "tunnel",   150,  (3, 3)),
        (16, "South Tunnel R",   "tunnel",   150,  (7, 3)),
        (17, "East Tunnel",      "tunnel",   200,  (9, 3)),
        (18, "West Tunnel",      "tunnel",   200,  (1, 3)),
        (19, "Exit North-West",  "exit",     500,  (1, 10)),
        (20, "Exit North-East",  "exit",     500,  (9, 10)),
        (21, "Exit South-West",  "exit",     600,  (1, 0)),
        (22, "Exit South-East",  "exit",     600,  (9, 0)),
        (23, "Exit East Main",   "exit",     800,  (10, 5)),
        (24, "Exit West Main",   "exit",     800,  (0, 5)),
    ]

    for nid, name, ntype, cap, pos in nodes:
        G.add_node(nid, name=name, type=ntype, capacity=cap,
                   current_crowd=0, risk=0.1, pos=pos)

    edges = [
        (1,9,1,1.0),(2,9,1,1.0),(3,9,1,1.0),
        (4,10,1,1.0),(5,10,1,1.0),(6,10,1,1.0),
        (7,11,1,1.0),(8,12,1,1.0),
        (9,13,1,0.5),(9,14,1,0.5),
        (10,15,1,0.5),(10,16,1,0.5),
        (11,17,1,0.5),(12,18,1,0.5),
        (13,19,1,2.0),(14,20,1,2.0),
        (15,21,1,2.0),(16,22,1,2.0),
        (17,23,1,2.0),(18,24,1,2.0),
        (9,11,2,1.0),(9,12,2,1.0),
        (10,11,2,1.0),(10,12,2,1.0),
        (11,9,2,1.0),(12,9,2,1.0),
        (9,1,1,1.0),(9,2,1,1.0),(9,3,1,1.0),
        (10,4,1,1.0),(10,5,1,1.0),(10,6,1,1.0),
    ]

    for u, v, dist, width in edges:
        G.add_edge(u, v, base_distance=dist, width_factor=width, weight=dist)

    return G


def update_edge_weights(G):
    for u, v, data in G.edges(data=True):
        risk_u   = G.nodes[u]['risk']
        risk_v   = G.nodes[v]['risk']
        cap_v    = G.nodes[v]['capacity']
        crowd_v  = G.nodes[v]['current_crowd']
        width    = data['width_factor']
        base     = data['base_distance']
        density  = min(crowd_v / cap_v if cap_v > 0 else 1.0, 1.0)
        congestion   = np.exp(3 * density) - 1
        risk_penalty = (risk_u + risk_v) * 5
        width_bonus  = 1.0 / width
        G[u][v]['weight'] = base + congestion + risk_penalty + width_bonus
    return G


def find_all_safe_exits(G, start_zone):
    G = update_edge_weights(G)
    results = []
    for exit_node in EXIT_NODES:
        try:
            path = nx.shortest_path(G, start_zone, exit_node, weight='weight')
            cost = nx.shortest_path_length(G, start_zone, exit_node, weight='weight')
            max_risk = max(G.nodes[n]['risk'] for n in path)
            results.append({
                'exit_id':       exit_node,
                'exit_name':     G.nodes[exit_node]['name'],
                'path':          path,
                'path_names':    [G.nodes[n]['name'] for n in path],
                'cost':          round(cost, 2),
                'max_risk':      round(max_risk, 2),
                'exit_capacity': G.nodes[exit_node]['capacity'],
                'safe':          max_risk < 0.6
            })
        except nx.NetworkXNoPath:
            continue
    results.sort(key=lambda x: x['cost'])
    return results


def get_best_exit(G, start_zone):
    routes = find_all_safe_exits(G, start_zone)
    if not routes: return None
    safe = [r for r in routes if r['safe']]
    return safe[0] if safe else routes[0]