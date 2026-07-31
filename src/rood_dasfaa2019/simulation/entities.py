from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Station:
    id: int
    x: float
    y: float

@dataclass(frozen=True)
class Order:
    id: int
    destination: int
    passengers: int
    arrival_time: float
    priority: float

@dataclass
class Bus:
    id: int
    capacity: int
    available_from: float
    depart_at: float
    route: list[int]
    station_travel_time: dict[int, float]
    accepted_passengers: int = 0
    accepted_time: float = 0.0
    accepted_orders: list[int] = field(default_factory=list)

    @property
    def remaining_seats(self) -> int:
        return self.capacity - self.accepted_passengers
