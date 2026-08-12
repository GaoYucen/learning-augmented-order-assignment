from __future__ import annotations
from .common import *

def ipd_dispatch(orders,buses,station_count,cfg):
    eps=float(cfg.get('epsilon',0.2)); buses=clone_buses(buses)
    sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}
    z={b.id:0.0 for b in buses}; u={b.id:0.0 for b in buses}; r={v:0.0 for v in range(station_count)}; q={v:0.0 for v in range(station_count)}; acc={}; reasons={}
    cand_stats=candidate_stats(orders,buses)
    for o in orders:
        cand=available_buses(o,buses)
        if not cand:
            reasons['no_candidate']=reasons.get('no_candidate',0)+1
            continue
        cand.sort(key=lambda b:o.passengers*z[b.id]+b.station_travel_time[o.destination]*o.passengers*(u[b.id]+q[o.destination]))
        b=cand[0]; tij=b.station_travel_time[o.destination]*o.passengers
        dual=o.passengers*(z[b.id]+r[o.destination])+tij*(u[b.id]+q[o.destination])
        reason=rejection_reason(o,b,sp,sx,sc,bt,st,dual)
        if reason!='accepted':
            reasons[reason]=reasons.get(reason,0)+1
            continue
        commit(o,b,sp,sx); acc[o.id]=b.id
        z[b.id]=z[b.id]*(1+o.passengers/max(b.capacity,1))+o.priority*eps/(4*max(b.capacity,1))
        r[o.destination]=r[o.destination]*(1+o.passengers/max(sc[o.destination],1e-9))+o.priority*eps/(4*max(sc[o.destination],1e-9))
        u[b.id]=u[b.id]*(1+tij/max(bt[b.id],1e-9))+o.priority*eps/(4*max(bt[b.id],1e-9))
        q[o.destination]=q[o.destination]*(1+tij/max(st[o.destination],1e-9))+o.priority*eps/(4*max(st[o.destination],1e-9))
    return SolverResult(summarize(acc,orders,buses), diagnostics=make_diagnostics(cand_stats,reasons))


def ipd_paper_dispatch(orders,buses,station_count,cfg):
    eps=float(cfg.get('epsilon',0.2)); buses=clone_buses(buses)
    sc,bt,st=limits(orders,buses,station_count,cfg); sp={v:0 for v in range(station_count)}; sx={v:0.0 for v in range(station_count)}
    z={b.id:0.0 for b in buses}; u={b.id:0.0 for b in buses}; r={v:0.0 for v in range(station_count)}; q={v:0.0 for v in range(station_count)}; acc={}; reasons={}
    cand_stats=candidate_stats(orders,buses)
    for o in orders:
        cand=available_buses(o,buses)
        if not cand:
            reasons['no_candidate']=reasons.get('no_candidate',0)+1
            continue
        cand.sort(key=lambda b:o.passengers*z[b.id]+b.station_travel_time[o.destination]*o.passengers*(u[b.id]+q[o.destination]))
        b=cand[0]; tij=b.station_travel_time[o.destination]*o.passengers
        dual=o.passengers*(z[b.id]+r[o.destination])+tij*(u[b.id]+q[o.destination])
        if dual>=o.priority:
            reasons['dual_price']=reasons.get('dual_price',0)+1
            continue
        commit(o,b,sp,sx); acc[o.id]=b.id
        z[b.id]=z[b.id]*(1+o.passengers/max(b.capacity,1))+o.priority*eps/(4*max(b.capacity,1))
        r[o.destination]=r[o.destination]*(1+o.passengers/max(sc[o.destination],1e-9))+o.priority*eps/(4*max(sc[o.destination],1e-9))
        u[b.id]=u[b.id]*(1+tij/max(bt[b.id],1e-9))+o.priority*eps/(4*max(bt[b.id],1e-9))
        q[o.destination]=q[o.destination]*(1+tij/max(st[o.destination],1e-9))+o.priority*eps/(4*max(st[o.destination],1e-9))
    return SolverResult(summarize(acc,orders,buses), diagnostics=make_diagnostics(cand_stats,reasons))
