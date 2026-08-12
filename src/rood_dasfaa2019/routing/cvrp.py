from __future__ import annotations
import math
import numpy as np
from rood_dasfaa2019.simulation.entities import Station


def build_bus_routes(stations: list[Station], demand, num_buses: int, capacity: int, cfg: dict | None = None, seed: int = 0) -> list[list[int]]:
    """Construct overlap-aware routes using weighted angular sweep.

    This is a deterministic selective-CVRP heuristic suitable for the manuscript's
    synthetic experiment setting.
    """
    cfg = cfg or {}
    rng=np.random.default_rng(seed)
    items=sorted(stations,key=lambda s:math.atan2(s.y,s.x))
    min_stops=int(cfg.get('route_min_stations',3))
    max_stops=max(min_stops,int(cfg.get('route_max_stations',5)))
    overlap=max(1,int(cfg.get('route_overlap_factor',3)))
    hot_boost=float(cfg.get('hot_station_boost',2.0))
    weights=np.array([max(float(demand[s.id]),1.0) for s in items],dtype=float)
    if len(weights):
        hot_idx=np.argsort(weights)[-max(1,len(weights)//5):]
        weights[hot_idx]*=hot_boost
        weights/=weights.sum()
    expanded=[]
    for s in items:
        copies=max(overlap,int(math.ceil(float(demand[s.id])*overlap/max(capacity,1))))
        expanded.extend([s.id]*copies)
    if not expanded:
        expanded=[s.id for s in items]
    routes=[]
    for j in range(num_buses):
        route=[]
        target=int(rng.integers(min_stops,max_stops+1))
        anchor=expanded[j % len(expanded)]
        route.append(anchor)
        while len(route)<target:
            sid=int(rng.choice([s.id for s in items],p=weights))
            if sid not in route:
                route.append(sid)
        route.sort(key=lambda sid: math.atan2(stations[sid].y, stations[sid].x))
        routes.append(route)
    # Ensure every hot station appears in multiple routes.
    for sid in np.argsort(weights)[-min(5,len(items)):]:
        covered=sum(1 for r in routes if items[int(sid)].id in r)
        needed=max(0, overlap-covered)
        for j in range(needed):
            r=routes[(int(sid)+j)%num_buses]
            if items[int(sid)].id not in r:
                if len(r) >= max_stops:
                    r[-1]=items[int(sid)].id
                else:
                    r.append(items[int(sid)].id)
    return routes
