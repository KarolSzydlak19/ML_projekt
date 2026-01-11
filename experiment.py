import os
import numpy as np
import random
from river.evaluate import progressive_val_score
from river import metrics, preprocessing, compose
from rich.progress import Progress, TimeElapsedColumn
import functools
import operator
from config import *

prog = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn()
)

combined_metric = functools.reduce(operator.add, METRICS.values())
prog.start()
for ds_name in DATASETS:
    prog_task = prog.add_task(f"Dataset: {ds_name}", total=len(METHODS))
    
    base_dataset = DATASETS[ds_name]
    base_dataset_list = list(base_dataset)
    
    for method in METHODS:
        #for seed in SEEDS:
        results_dir = os.path.join(RESULTS_DIR, ds_name)
        os.makedirs(results_dir, exist_ok=True)
            #shuffled_data = base_dataset_list.copy()
            #rng = random.Random(seed)
            #rng.shuffle(shuffled_data)
            #stream_dataset = iter(shuffled_data)
        model_pipeline = (
            (compose.SelectType(str) | preprocessing.OneHotEncoder()) +
            (compose.SelectType(int, float) | preprocessing.StandardScaler())
        ) | METHODS[method].clone()
        multi_metric = progressive_val_score(
            dataset=iter(base_dataset_list),
            model=model_pipeline,
            metric=combined_metric.clone()
        )
        current_result = [{
            "dataset_name": ds_name,
            "method_name": method,
            "metrics": multi_metric.get()
        }]

        prog.advance(prog_task)
        np.save(os.path.join(results_dir, f"{method}.npy"), current_result)
prog.advance(prog_task)
prog.stop()
