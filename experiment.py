import os
import numpy as np
import random
from river.evaluate import iter_progressive_val_score
from river import metrics, preprocessing, compose
from rich.progress import Progress, TimeElapsedColumn
import functools
import operator
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import random
from config import *
matplotlib.use('Agg')
prog = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn()
)

#combined_metric = functools.reduce(operator.add, METRICS.values())
prog.start()
for ds_name in DATASETS:
    prog_task = prog.add_task(f"Dataset: {ds_name}", total=len(METHODS))
    
    
    for method in METHODS:
        #for seed in SEEDS:
        results_dir = os.path.join(RESULTS_DIR, ds_name)
        os.makedirs(results_dir, exist_ok=True)

        #base_dataset_list_all = DATASETS[ds_name]
        base_dataset_list_all = DATASETS[ds_name]()
        all_histories = {m_name: [] for m_name in METRICS.keys()}
        for run_id, base_dataset in enumerate(base_dataset_list_all):
            model_pipeline = (
            compose.SelectType(int, float) 
            ) | METHODS[method].clone()
            current_metric = {
                name: metric.clone() 
                for name, metric in METRICS.items()
            }
            history = {m_name: [] for m_name in METRICS.keys()}
            steps = []
            base_dataset_list = list(base_dataset)
            for i, (x,y) in enumerate(base_dataset_list):
                y_pred = model_pipeline.predict_one(x)
                model_pipeline.learn_one(x, y)

                for m_name, metric in current_metric.items():
                    if i > WARMUP and y_pred is not None:
                        metric.update(y, y_pred)
                        score = metric.get()
                        if y_pred is None or y_pred == -1:
                            score = np.nan
                        history[m_name].append(score)
                    else:
                        history[m_name].append(np.nan)
                steps.append(i)

            for m_name in METRICS.keys():
                all_histories[m_name].append(history[m_name])

        mean_history = {m_name: np.nanmean(all_histories[m_name], axis=0) for m_name in METRICS.keys()}
        time_now = datetime.now().strftime("%d%m%Y%H%M%S")
        os.makedirs(os.path.join(PLOT_DIR, ds_name, method), exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, ds_name, method, f"{time_now}.png")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(steps, mean_history['AdjustedRand'], label='AdjustedRandIndex')
        ax.plot(steps, mean_history['NormalizedMutualInfo'], label='NormalizedMutualInfo')
        ax.plot(steps, mean_history['Homogeneity'], label='Homogeneity')
        #ax.set_ylim(-1, 1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title(f"{ds_name} – {method}")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(plot_path)
        plt.close(fig)
        
        final_scores = {name: m.get() for name, m in current_metric.items()}
                    
        current_result = [{
            "dataset_name": ds_name,
            "method_name": method,
            "steps": steps,
            "history": history,
            "final_metrics": final_scores
        }]

        prog.advance(prog_task)
        np.save(os.path.join(results_dir, f"{method}.npy"), current_result)
prog.advance(prog_task)
prog.stop()
