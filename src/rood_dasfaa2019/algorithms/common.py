from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any
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
    solver_dual_bound: float | None = None
    solver_mip_gap: float | None = None
    solver_node_count: int | None = None
    request_latencies_ms: list[float] | None = None

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
    if 'station_capacity' in cfg and 'bus_time_capacity' in cfg and 'station_time_capacity' in cfg:
        return (
            dict(cfg['station_capacity']),
            dict(cfg['bus_time_capacity']),
            dict(cfg['station_time_capacity']),
        )
    total_capacity=sum(b.capacity for b in buses)
    demand=[0]*station_count
    for o in orders: demand[o.destination]+=o.passengers
    total=max(sum(demand),1)
    explicit_station_cap = cfg.get('station_capacity')
    if explicit_station_cap is not None:
        station_cap = _indexed_values(explicit_station_cap, range(station_count), 'station_capacity')
    else:
        station_cap={v:max(1.0,total_capacity*demand[v]/total*cfg.get('station_fairness_scale',1.0)) for v in range(station_count)}
    avg_t=sum(t for b in buses for t in b.station_travel_time.values())/max(sum(len(b.station_travel_time) for b in buses),1)
    explicit_bus_time = cfg.get('bus_time_capacity')
    if explicit_bus_time is not None:
        bus_time = _indexed_values(explicit_bus_time, [b.id for b in buses], 'bus_time_capacity')
    else:
        bus_time={
            b.id: b.capacity * (sum(b.station_travel_time.values())/max(len(b.station_travel_time),1)) * cfg.get('bus_time_scale',1.0)
            for b in buses
        }
    explicit_station_time = cfg.get('station_time_capacity')
    if explicit_station_time is not None:
        station_time = _indexed_values(explicit_station_time, range(station_count), 'station_time_capacity')
    else:
        station_time={
            v: station_cap[v] * source_station_time(v, buses, avg_t) * cfg.get('station_time_scale',1.0)
            for v in range(station_count)
        }
    return station_cap,bus_time,station_time


def _indexed_values(values: Any, keys, name: str) -> dict:
    if isinstance(values, dict):
        result = {}
        for key in keys:
            if key in values:
                result[key] = float(values[key])
            elif str(key) in values:
                result[key] = float(values[str(key)])
            else:
                raise KeyError(f'{name} is missing key {key}')
        return result
    sequence = list(values)
    result = {}
    for key in keys:
        if int(key) >= len(sequence):
            raise IndexError(f'{name} is missing index {key}')
        result[key] = float(sequence[int(key)])
    return result


def count_constraint_violations(accepted, orders, buses, station_count, cfg, tolerance: float = 1e-8) -> int:
    order_map = {order.id: order for order in orders}
    bus_map = {bus.id: bus for bus in buses}
    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    bus_passengers = {bus.id: 0.0 for bus in buses}
    bus_used_time = {bus.id: 0.0 for bus in buses}
    station_passengers = {station: 0.0 for station in range(station_count)}
    station_used_time = {station: 0.0 for station in range(station_count)}
    violations = 0
    for order_id, bus_id in accepted.items():
        order = order_map[order_id]
        bus = bus_map[bus_id]
        if not (bus.available_from <= order.arrival_time <= bus.depart_at and order.destination in bus.route):
            violations += 1
            continue
        travel = order.passengers * bus.station_travel_time[order.destination]
        bus_passengers[bus_id] += order.passengers
        bus_used_time[bus_id] += travel
        station_passengers[order.destination] += order.passengers
        station_used_time[order.destination] += travel
    violations += sum(bus_passengers[bus.id] > bus.capacity + tolerance for bus in buses)
    violations += sum(bus_used_time[bus.id] > bus_time[bus.id] + tolerance for bus in buses)
    violations += sum(station_passengers[station] > station_cap[station] + tolerance for station in range(station_count))
    violations += sum(station_used_time[station] > station_time[station] + tolerance for station in range(station_count))
    return int(violations)


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
