#!/usr/bin/env python
import argparse, json
from rood_dasfaa2019.utils.config import load_experiment
from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.algorithms import random_dispatch,greedy_dispatch,paper_random_dispatch,paper_greedy_dispatch,fpd_dispatch,ipd_dispatch,ipd_paper_dispatch,offline_opt
from rood_dasfaa2019.algorithms.common import candidate_stats

def enrich(result):
 if hasattr(result,'dispatch'):
  base=vars(result.dispatch).copy(); accepted=base.pop('accepted')
  base['orders']=len(accepted); base['avg_travel_time']=base['travel_time']/max(base['passengers'],1)
  if result.diagnostics is not None: base['diagnostics']=vars(result.diagnostics)
  if result.solver_status is not None:
   base['solver_status']=result.solver_status; base['solver_success']=result.solver_success; base['solver_time_limit_hit']=result.solver_time_limit_hit; base['solver_message']=result.solver_message
  return base
 d=vars(result).copy(); accepted=d.pop('accepted');
 d['orders']=len(accepted); d['avg_travel_time']=d['travel_time']/max(d['passengers'],1); return d

def main():
 p=argparse.ArgumentParser(); p.add_argument('--experiment',type=int,default=1,choices=range(1,6)); p.add_argument('--seed',type=int,default=2019); p.add_argument('--skip-opt',action='store_true'); a=p.parse_args()
 cfg=load_experiment(a.experiment); st,orders,buses=generate_instance(cfg,a.seed+a.experiment)
 algs={'PaperRandom':lambda:paper_random_dispatch(orders,buses,len(st),cfg,a.seed),'PaperGreedy':lambda:paper_greedy_dispatch(orders,buses,len(st),cfg),'RandomFF':lambda:random_dispatch(orders,buses,len(st),cfg,a.seed),'GreedyFF':lambda:greedy_dispatch(orders,buses,len(st),cfg),'FPD':lambda:fpd_dispatch(orders,buses,len(st),cfg),'IPD-paper':lambda:ipd_paper_dispatch(orders,buses,len(st),cfg),'IPD-feasible':lambda:ipd_dispatch(orders,buses,len(st),cfg)}
 if not a.skip_opt: algs['OPT']=lambda:offline_opt(orders,buses,len(st),cfg)
 out={k:enrich(fn()) for k,fn in algs.items()}
 out['_diagnostics']=vars(candidate_stats(orders,buses))
 if 'OPT' in out:
  opt=out['OPT']
  for k,v in out.items():
   if k=='_diagnostics': continue
   v['priority_ratio_to_opt']=v['objective']/max(opt['objective'],1e-9)
   v['passenger_ratio_to_opt']=v['passengers']/max(opt['passengers'],1e-9)
   v['orders_ratio_to_opt']=v['orders']/max(opt['orders'],1e-9)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
