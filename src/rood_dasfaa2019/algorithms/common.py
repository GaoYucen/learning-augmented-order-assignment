from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from rood_dasfaa2019.simulation.entities import Order, Bus

@dataclass
class DispatchResult:
    accepted: dict[int,int]
    objective: float
    passengers: int
    travel_time: float


def available_buses(order: Order, buses: list[Bus]) -> list[Bus]:
    return [b for b in buses if b.available_from <= order.arrival_time <= b.depart_at and order.destination in b.route]


def limits(orders: list[Order], buses: list[Bus], station_count: int, cfg: dict):
    total_capacity=sum(b.capacity for b in buses)
    demand=[0]*station_count
    for o in orders: demand[o.destination]+=o.passengers
    total=max(sum(demand),1)
    station_cap={v:max(1.0,total_capacity*demand[v]/total*cfg.get('station_fairness_scale',1.0)) for v in range(station_count)}
    avg_t=sum(t for b in buses for t in b.station_travel_time.values())/max(sum(len(b.station_travel_time) for b in buses),1)
    bus_time={b.id:b.capacity*avg_t*cfg.get('bus_time_scale',1.0) for b in buses}
    station_time={v:station_cap[v]*max([b.station_travel_time.get(v,avg_t) for b in buses])*cfg.get('station_time_scale',1.0) for v in range(station_count)}
    return station_cap,bus_time,station_time


def feasible(o,b,st_pass,st_time,station_cap,bus_time,station_time):
    t=b.station_travel_time[o.destination]*o.passengers
    return (b.accepted_passengers+o.passengers <= b.capacity and
            st_pass[o.destination]+o.passengers <= station_cap[o.destination]+1e-9 and
            b.accepted_time+t <= bus_time[b.id]+1e-9 and
            st_time[o.destination]+t <= station_time[o.destination]+1e-9)

def commit(o,b,st_pass,st_time):
    t=b.station_travel_time[o.destination]*o.passengers
    b.accepted_passengers+=o.passengers; b.accepted_time+=t; b.accepted_orders.append(o.id)
    st_pass[o.destination]+=o.passengers; st_time[o.destination]+=t

def summarize(accepted,orders,buses):
    om={o.id:o for o in orders}; obj=sum(om[i].priority for i in accepted)
    passengers=sum(om[i].passengers for i in accepted)
    travel=sum(om[i].passengers*next(b.station_travel_time[om[i].destination] for b in buses if b.id==j) for i,j in accepted.items())
    return DispatchResult(accepted,obj,passengers,travel)

def clone_buses(buses): return deepcopy(buses)
