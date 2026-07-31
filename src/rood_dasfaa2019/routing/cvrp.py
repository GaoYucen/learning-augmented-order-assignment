from __future__ import annotations
import math
from rood_dasfaa2019.simulation.entities import Station


def build_bus_routes(stations: list[Station], demand, num_buses: int, capacity: int) -> list[list[int]]:
    """Construct capacity-aware routes using weighted angular sweep.

    This is a deterministic selective-CVRP heuristic suitable for the manuscript's
    synthetic experiment setting.
    """
    items=sorted(stations,key=lambda s:math.atan2(s.y,s.x))
    expanded=[]
    for s in items:
        copies=max(1,int(math.ceil(float(demand[s.id])/max(capacity,1))))
        expanded.extend([s.id]*copies)
    routes=[[] for _ in range(num_buses)]
    for k,sid in enumerate(expanded):
        r=routes[k%num_buses]
        if sid not in r: r.append(sid)
    # Ensure every bus has at least one feasible station.
    for j,r in enumerate(routes):
        if not r: r.append(items[j%len(items)].id)
    return routes
