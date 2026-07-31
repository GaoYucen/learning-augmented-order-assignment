from __future__ import annotations
import math
from .common import *

def fpd_dispatch(orders,buses,station_count,cfg,sigma=0.5,step=0.05):
    """Discretized simulation of the paper's continuous FPD updates."""
    buses=clone_buses(buses); alpha=-math.log(sigma); beta=(math.log(sigma)**2)/8
    sc,bt,st=limits(orders,buses,station_count,cfg); x={}; z={b.id:0.0 for b in buses}; u={b.id:0.0 for b in buses}; r={v:0.0 for v in range(station_count)}; q={v:0.0 for v in range(station_count)}
    for o in orders:
        cand=available_buses(o,buses)
        for b in cand:
            key=(o.id,b.id); x[key]=0.0; tij=b.station_travel_time[o.destination]*o.passengers
            target=o.priority*(1-math.exp(-alpha))
            for _ in range(int(1/step)):
                yi=o.priority*math.exp(alpha*(sum(x.get((o.id,k.id),0) for k in cand)-1))-o.priority*math.exp(-alpha)
                if yi+o.passengers*(z[b.id]+r[o.destination])+tij*(u[b.id]+q[o.destination])>=target: break
                x[key]=min(1.0,x[key]+step)
                z[b.id]=max(z[b.id],(o.priority/o.passengers)*(math.exp(beta*x[key]*o.passengers/(2*b.capacity))-1))
                r[o.destination]=max(r[o.destination],(o.priority/o.passengers)*(math.exp(beta*x[key]*o.passengers/(2*sc[o.destination]))-1))
                u[b.id]=max(u[b.id],(o.priority/max(tij,1e-9))*(math.exp(beta*x[key]*tij/(2*bt[b.id]))-1))
                q[o.destination]=max(q[o.destination],(o.priority/max(tij,1e-9))*(math.exp(beta*x[key]*tij/(2*st[o.destination]))-1))
    objective=sum(next(o.priority for o in orders if o.id==i)*val for (i,j),val in x.items())
    passengers=sum(next(o.passengers for o in orders if o.id==i)*val for (i,j),val in x.items())
    travel=sum(next(o.passengers for o in orders if o.id==i)*next(b.station_travel_time[next(o.destination for o in orders if o.id==i)] for b in buses if b.id==j)*val for (i,j),val in x.items())
    return DispatchResult({i:j for (i,j),v in x.items() if v>0},objective,passengers,travel)
