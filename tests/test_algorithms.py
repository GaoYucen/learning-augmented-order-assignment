from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.algorithms import random_dispatch,greedy_dispatch,ipd_dispatch

def cfg(): return dict(num_stations=8,horizon_minutes=60,bus_wait_minutes=15,epsilon=.2,max_passengers_per_order=3,station_fairness_scale=1,bus_time_scale=1,station_time_scale=1,num_buses=12,bus_capacity=15,num_orders=100)

def test_online_methods_run():
 c=cfg(); s,o,b=generate_instance(c,7)
 for fn in (random_dispatch,greedy_dispatch,ipd_dispatch):
  r=fn(o,b,len(s),c)
  assert r.objective>=0 and r.passengers>=0
  assert len(r.accepted)<=len(o)
