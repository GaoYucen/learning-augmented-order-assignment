from pathlib import Path
import yaml

def load_experiment(experiment:int,path='configs/experiments.yaml'):
    data=yaml.safe_load(Path(path).read_text())
    cfg=dict(data['common']); cfg.update(data['experiments'][experiment]); return cfg
