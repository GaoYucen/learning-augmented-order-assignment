from __future__ import annotations
import math
import numpy as np
from .entities import Station, Order, Bus
from rood_dasfaa2019.routing.cvrp import build_bus_routes


def generate_stations(n: int = 30) -> list[Station]:
    # Deterministic radial benchmark around an airport at (0, 0).
    stations=[]
    golden=math.pi*(3-math.sqrt(5))
    for i in range(n):
        r=4.0+0.65*math.sqrt(i+1)
        a=i*golden
        stations.append(Station(i, r*math.cos(a), r*math.sin(a)))
    return stations


def generate_instance(cfg: dict, seed: int):
    rng=np.random.default_rng(seed)
    stations=generate_stations(cfg['num_stations'])
    weights=np.linspace(1.8,0.6,len(stations)); weights/=weights.sum()
    orders=[]
    for i in range(cfg['num_orders']):
        orders.append(Order(
            id=i,
            destination=int(rng.choice(len(stations), p=weights)),
            passengers=int(rng.integers(1, cfg['max_passengers_per_order']+1)),
            arrival_time=float(rng.uniform(0,cfg['horizon_minutes'])),
            priority=float(rng.uniform(0.01,1.0)),
        ))
    orders.sort(key=lambda o:(o.arrival_time,o.id))
    horizon=cfg['horizon_minutes']-cfg['bus_wait_minutes']
    if cfg.get('bus_arrival_mode','uniform')=='wave':
        wave_count=max(1,int(cfg.get('bus_wave_count',3)))
        wave_std=float(cfg.get('bus_wave_std_minutes',8))
        centers=np.linspace(horizon*0.2,horizon*0.8,wave_count)
        picks=rng.choice(centers,size=cfg['num_buses'])
        bus_arrivals=np.clip(rng.normal(picks,wave_std),0,horizon)
        bus_arrivals=np.sort(bus_arrivals)
    else:
        bus_arrivals=np.sort(rng.uniform(0,horizon,cfg['num_buses']))
    demand=np.bincount([o.destination for o in orders], minlength=len(stations)).astype(float)
    noise=float(cfg.get('prediction_noise',0.25))
    predicted=np.maximum(1.0,demand*rng.lognormal(mean=0.0,sigma=noise,size=len(demand)))
    routes=build_bus_routes(stations,predicted,cfg['num_buses'],cfg['bus_capacity'],cfg=cfg,seed=seed)
    buses=[]
    for j,t in enumerate(bus_arrivals):
        route=routes[j]
        elapsed=0.0; times={}
        prev=(0.0,0.0)
        for sid in route:
            s=stations[sid]
            elapsed += math.dist(prev,(s.x,s.y))*3.0
            times[sid]=elapsed
            prev=(s.x,s.y)
        buses.append(Bus(j,cfg['bus_capacity'],float(t),float(t+cfg['bus_wait_minutes']),route,times))
    return stations,orders,buses
