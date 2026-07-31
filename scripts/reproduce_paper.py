#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from rood_dasfaa2019.utils.config import load_experiment
from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.algorithms import random_dispatch,greedy_dispatch,ipd_dispatch,offline_opt

def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=2019); p.add_argument('--skip-opt',action='store_true'); a=p.parse_args(); rows=[]
 for e in range(1,6):
  cfg=load_experiment(e); st,orders,buses=generate_instance(cfg,a.seed+e)
  methods={'Random':random_dispatch(orders,buses,len(st),cfg,a.seed+e),'Greedy':greedy_dispatch(orders,buses,len(st),cfg),'IPD':ipd_dispatch(orders,buses,len(st),cfg)}
  if not a.skip_opt: methods['OPT']=offline_opt(orders,buses,len(st),cfg)
  for name,r in methods.items(): rows.append({'experiment':e,'method':name,'orders':len(r.accepted),'passengers':r.passengers,'priority':r.objective,'travel_time':r.travel_time})
 out=Path('outputs'); out.mkdir(exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'results.csv',index=False)
 for metric in ['orders','passengers','priority']:
  pivot=df.pivot(index='experiment',columns='method',values=metric); ax=pivot.plot(kind='bar'); ax.set_ylabel(metric.replace('_',' ').title()); ax.figure.tight_layout(); ax.figure.savefig(out/f'{metric}.png',dpi=180); plt.close(ax.figure)
 print(df.to_string(index=False)); print(f'Wrote {out / "results.csv"}')
if __name__=='__main__':main()
