from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
from rood_dasfaa2019.simulation.entities import Order, Bus

@dataclass
class DispatchResult:
    accepted: dict[int,int]
    objective: float
    passengers: int
    travel_time: float


@dataclass
class CandidateStats:
    average_candidates: float
    zero_candidate_orders: int
    zero_candidate_rate: float
    max_candidates: int


@dataclass
class DiagnosticStats:
    average_candidates: float
    zero_candidate_orders: int
    zero_candidate_rate: float
    max_candidates: int
    rejected_no_candidate: int
    rejected_capacity: int
    rejected_station_cap: int
    rejected_bus_time: int
    rejected_station_time: int
    rejected_dual_price: int


@dataclass
class SolverResult:
    dispatch: DispatchResult
    diagnostics: DiagnosticStats | None = None
    solver_status: str | None = None
    solver_success: bool | None = None
    solver_time_limit_hit: bool | None = None
    solver_message: str | None = None

    @property
    def accepted(self):
        return self.dispatch.accepted

    @property
    def objective(self):
        return self.dispatch.objective

    @property
    def passengers(self):
        return self.dispatch.passengers

    @property
    def travel_time(self):
        return self.dispatch.travel_time


def available_buses(order: Order, buses: list[Bus]) -> list[Bus]:
    return [b for b in buses if b.available_from <= order.arrival_time <= b.depart_at and order.destination in b.route]


def candidate_stats(orders: list[Order], buses: list[Bus]) -> CandidateStats:
    counts=[len(available_buses(o,buses)) for o in orders]
    if not counts:
        return CandidateStats(0.0,0,0.0,0)
    zero=sum(1 for c in counts if c==0)
    return CandidateStats(sum(counts)/len(counts),zero,zero/len(counts),max(counts))


def source_station_time(station_id: int, buses: list[Bus], default: float = 0.0) -> float:
    vals=[b.station_travel_time[station_id] for b in buses if station_id in b.station_travel_time]
    return sum(vals)/len(vals) if vals else default


def limits(orders: list[Order], buses: list[Bus], station_count: int, cfg: dict):
    total_capacity=sum(b.capacity for b in buses)
    demand=[0]*station_count
    for o in orders: demand[o.destination]+=o.passengers
    total=max(sum(demand),1)
    station_cap={v:max(1.0,total_capacity*demand[v]/total*cfg.get('station_fairness_scale',1.0)) for v in range(station_count)}
    avg_t=sum(t for b in buses for t in b.station_travel_time.values())/max(sum(len(b.station_travel_time) for b in buses),1)
    bus_time={
        b.id: b.capacity * (sum(b.station_travel_time.values())/max(len(b.station_travel_time),1)) * cfg.get('bus_time_scale',1.0)
        for b in buses
    }
    station_time={
        v: station_cap[v] * source_station_time(v, buses, avg_t) * cfg.get('station_time_scale',1.0)
        for v in range(station_count)
    }
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


def rejection_reason(o,b,st_pass,st_time,station_cap,bus_time,station_time,dual_value=None):
    if b is None:
        return 'no_candidate'
    t=b.station_travel_time[o.destination]*o.passengers
    if b.accepted_passengers+o.passengers > b.capacity:
        return 'capacity'
    if st_pass[o.destination]+o.passengers > station_cap[o.destination]+1e-9:
        return 'station_cap'
    if b.accepted_time+t > bus_time[b.id]+1e-9:
        return 'bus_time'
    if st_time[o.destination]+t > station_time[o.destination]+1e-9:
        return 'station_time'
    if dual_value is not None and dual_value >= o.priority:
        return 'dual_price'
    return 'accepted'


def make_diagnostics(candidate: CandidateStats, reasons: dict[str, int]) -> DiagnosticStats:
    return DiagnosticStats(
        average_candidates=candidate.average_candidates,
        zero_candidate_orders=candidate.zero_candidate_orders,
        zero_candidate_rate=candidate.zero_candidate_rate,
        max_candidates=candidate.max_candidates,
        rejected_no_candidate=reasons.get('no_candidate',0),
        rejected_capacity=reasons.get('capacity',0),
        rejected_station_cap=reasons.get('station_cap',0),
        rejected_bus_time=reasons.get('bus_time',0),
        rejected_station_time=reasons.get('station_time',0),
        rejected_dual_price=reasons.get('dual_price',0),
    )
