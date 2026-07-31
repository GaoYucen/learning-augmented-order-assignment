from __future__ import annotations
import numpy as np
from .common import *

def random_dispatch(orders,buses,station_count,cfg,seed=0):
    rng=np.random.default_rng(seed); buses=clone_buses(buses)
    sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}; acc={}
    for o in orders:
        cand=available_buses(o,buses); rng.shuffle(cand)
        for b in cand:
            if feasible(o,b,sp,sx,sc,bt,st): commit(o,b,sp,sx); acc[o.id]=b.id; break
    return summarize(acc,orders,buses)

def greedy_dispatch(orders,buses,station_count,cfg):
    buses=clone_buses(buses); sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}; acc={}
    for o in orders:
        cand=sorted(available_buses(o,buses),key=lambda b:(-b.remaining_seats,b.station_travel_time[o.destination]))
        for b in cand:
            if feasible(o,b,sp,sx,sc,bt,st): commit(o,b,sp,sx); acc[o.id]=b.id; break
    return summarize(acc,orders,buses)
