from __future__ import annotations
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from .common import DispatchResult, SolverResult, available_buses, limits

def offline_opt(orders,buses,station_count,cfg):
    pairs=[]
    for o in orders:
        for b in available_buses(o,buses): pairs.append((o,b))
    if not pairs: return SolverResult(DispatchResult({},0.0,0,0.0), solver_status='no_pairs', solver_success=True, solver_time_limit_hit=False, solver_message='No feasible order-bus pairs')
    sc,bt,st=limits(orders,buses,station_count,cfg)
    rows=[]; ub=[]
    # each order at most one bus
    for o in orders:
        idx=[k for k,(oo,b) in enumerate(pairs) if oo.id==o.id]
        if idx: rows.append(idx); ub.append(1.0)
    # bus capacity and bus time
    for b in buses:
        idx=[k for k,(o,bb) in enumerate(pairs) if bb.id==b.id]
        if idx:
            rows.append([(k,pairs[k][0].passengers) for k in idx]); ub.append(b.capacity)
            rows.append([(k,pairs[k][0].passengers*b.station_travel_time[pairs[k][0].destination]) for k in idx]); ub.append(bt[b.id])
    # station passenger/time
    for v in range(station_count):
        idx=[k for k,(o,b) in enumerate(pairs) if o.destination==v]
        if idx:
            rows.append([(k,pairs[k][0].passengers) for k in idx]); ub.append(sc[v])
            rows.append([(k,pairs[k][0].passengers*pairs[k][1].station_travel_time[v]) for k in idx]); ub.append(st[v])
    A=lil_matrix((len(rows),len(pairs)))
    for i,row in enumerate(rows):
        if row and isinstance(row[0],tuple):
            for k,val in row:A[i,k]=val
        else:
            for k in row:A[i,k]=1
    c=-np.array([o.priority for o,b in pairs])
    time_limit=float(cfg.get('opt_time_limit_seconds',30))
    res=milp(c,integrality=np.ones(len(pairs)),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),-np.inf,np.array(ub)),options={'time_limit':time_limit})
    x=np.zeros(len(pairs)) if res.x is None else res.x
    acc={pairs[k][0].id:pairs[k][1].id for k,val in enumerate(x) if val>0.5}
    obj=sum(pairs[k][0].priority for k,val in enumerate(x) if val>0.5)
    pas=sum(pairs[k][0].passengers for k,val in enumerate(x) if val>0.5)
    tt=sum(pairs[k][0].passengers*pairs[k][1].station_travel_time[pairs[k][0].destination] for k,val in enumerate(x) if val>0.5)
    return SolverResult(
        DispatchResult(acc,obj,pas,tt),
        solver_status=str(getattr(res,'status',None)),
        solver_success=bool(getattr(res,'success',False)),
        solver_time_limit_hit='time limit' in str(getattr(res,'message','')).lower(),
        solver_message=str(getattr(res,'message','')),
        solver_dual_bound=float(getattr(res,'mip_dual_bound',np.nan)) if getattr(res,'mip_dual_bound',None) is not None else None,
        solver_mip_gap=float(getattr(res,'mip_gap',np.nan)) if getattr(res,'mip_gap',None) is not None else None,
        solver_node_count=int(getattr(res,'mip_node_count',0)) if getattr(res,'mip_node_count',None) is not None else None,
    )
