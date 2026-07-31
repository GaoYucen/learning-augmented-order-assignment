#!/usr/bin/env python
import argparse, json
from rood_dasfaa2019.utils.config import load_experiment
from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.algorithms import random_dispatch,greedy_dispatch,fpd_dispatch,ipd_dispatch,offline_opt

def main():
 p=argparse.ArgumentParser(); p.add_argument('--experiment',type=int,default=1,choices=range(1,6)); p.add_argument('--seed',type=int,default=2019); p.add_argument('--skip-opt',action='store_true'); a=p.parse_args()
 cfg=load_experiment(a.experiment); st,orders,buses=generate_instance(cfg,a.seed+a.experiment)
 algs={'Random':lambda:random_dispatch(orders,buses,len(st),cfg,a.seed),'Greedy':lambda:greedy_dispatch(orders,buses,len(st),cfg),'FPD':lambda:fpd_dispatch(orders,buses,len(st),cfg),'IPD':lambda:ipd_dispatch(orders,buses,len(st),cfg)}
 if not a.skip_opt: algs['OPT']=lambda:offline_opt(orders,buses,len(st),cfg)
 out={k:vars(fn()) for k,fn in algs.items()}
 for v in out.values(): v['orders']=len(v.pop('accepted'))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
