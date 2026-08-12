from __future__ import annotations
import numpy as np
from .common import *

def paper_random_dispatch(orders,buses,station_count,cfg,seed=0):
    rng=np.random.default_rng(seed); buses=clone_buses(buses)
    sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}; acc={}; reasons={}; cand_stats=candidate_stats(orders,buses)
    for o in orders:
        cand=available_buses(o,buses)
        if not cand: reasons['no_candidate']=reasons.get('no_candidate',0)+1; continue
        b=rng.choice(cand)
        reason=rejection_reason(o,b,sp,sx,sc,bt,st)
        if reason!='accepted': reasons[reason]=reasons.get(reason,0)+1; continue
        commit(o,b,sp,sx); acc[o.id]=b.id
    return SolverResult(summarize(acc,orders,buses), diagnostics=make_diagnostics(cand_stats,reasons))

def paper_greedy_dispatch(orders,buses,station_count,cfg):
    buses=clone_buses(buses); sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}; acc={}; reasons={}; cand_stats=candidate_stats(orders,buses)
    for o in orders:
        cand=available_buses(o,buses)
        if not cand: reasons['no_candidate']=reasons.get('no_candidate',0)+1; continue
        b=max(cand,key=lambda b:b.remaining_seats)
        reason=rejection_reason(o,b,sp,sx,sc,bt,st)
        if reason!='accepted': reasons[reason]=reasons.get(reason,0)+1; continue
        commit(o,b,sp,sx); acc[o.id]=b.id
    return SolverResult(summarize(acc,orders,buses), diagnostics=make_diagnostics(cand_stats,reasons))

def random_dispatch(orders,buses,station_count,cfg,seed=0):
    rng=np.random.default_rng(seed); buses=clone_buses(buses)
    sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}; acc={}; reasons={}; cand_stats=candidate_stats(orders,buses)
    for o in orders:
        cand=available_buses(o,buses); rng.shuffle(cand)
        if not cand: reasons['no_candidate']=reasons.get('no_candidate',0)+1; continue
        accepted=False
        for b in cand:
            reason=rejection_reason(o,b,sp,sx,sc,bt,st)
            if reason=='accepted': commit(o,b,sp,sx); acc[o.id]=b.id; accepted=True; break
        if not accepted:
            reasons[rejection_reason(o,cand[0],sp,sx,sc,bt,st)]=reasons.get(rejection_reason(o,cand[0],sp,sx,sc,bt,st),0)+1
    return SolverResult(summarize(acc,orders,buses), diagnostics=make_diagnostics(cand_stats,reasons))

def greedy_dispatch(orders,buses,station_count,cfg):
    buses=clone_buses(buses); sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}; acc={}; reasons={}; cand_stats=candidate_stats(orders,buses)
    for o in orders:
        cand=sorted(available_buses(o,buses),key=lambda b:(-b.remaining_seats,b.station_travel_time[o.destination]))
        if not cand: reasons['no_candidate']=reasons.get('no_candidate',0)+1; continue
        accepted=False
        for b in cand:
            reason=rejection_reason(o,b,sp,sx,sc,bt,st)
            if reason=='accepted': commit(o,b,sp,sx); acc[o.id]=b.id; accepted=True; break
        if not accepted:
            reasons[rejection_reason(o,cand[0],sp,sx,sc,bt,st)]=reasons.get(rejection_reason(o,cand[0],sp,sx,sc,bt,st),0)+1
    return SolverResult(summarize(acc,orders,buses), diagnostics=make_diagnostics(cand_stats,reasons))
