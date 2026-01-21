import os
import numpy as np
from river import utils, compose
from rich.progress import Progress, TimeElapsedColumn
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

from config import * 
matplotlib.use('Agg')

prog = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn()
)

prog.start()

for ds_name in DATASETS:
    prog_task = prog.add_task(f"Dataset: {ds_name}", total=len(METHODS))
    
    for method in METHODS:
        results_dir = os.path.join(RESULTS_DIR, ds_name)
        os.makedirs(results_dir, exist_ok=True)

        base_dataset_list_all = DATASETS[ds_name]()
        
        all_histories = {m_name: [] for m_name in METRICS.keys()}

        for run_id, base_dataset in enumerate(base_dataset_list_all):
            
            model_pipeline = (
                compose.SelectType(int, float) 
            ) | METHODS[method].clone()
            
            current_metric = {
                name: utils.Rolling(metric.clone(), window_size=WINDOW_SIZE) 
                for name, metric in METRICS.items()
            }
            
            history = {m_name: [] for m_name in METRICS.keys()}
            steps = []
            
            for i, (x, y) in enumerate(base_dataset):
                
                y_pred = model_pipeline.predict_one(x)
                model_pipeline.learn_one(x, y)

                if y_pred is not None:
                    for m_name, metric in current_metric.items():
                        metric.update(y, y_pred)
                        
                        if i > WARMUP:
                            score = metric.get()
                            history[m_name].append(score if score is not None else np.nan)
                        else:
                            history[m_name].append(np.nan)
                else:
                    for m_name in METRICS.keys():
                        history[m_name].append(np.nan)
                
                steps.append(i)

            for m_name in METRICS.keys():
                all_histories[m_name].append(history[m_name])

        mean_history = {
            m_name: np.nanmean(all_histories[m_name], axis=0) 
            for m_name in METRICS.keys()
        }
        
        # --- Rysowanie Wykresu ---
        time_now = datetime.now().strftime("%d%m%Y%H%M%S")
        plot_dir_full = os.path.join(PLOT_DIR, ds_name, method)
        os.makedirs(plot_dir_full, exist_ok=True)
        plot_path = os.path.join(plot_dir_full, f"{time_now}.png")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for m_name, m_values in mean_history.items():
            ax.plot(steps, m_values, label=m_name)

        ax.set_xlabel("Step")
        ax.set_ylabel(f"Score (Rolling Window: {WINDOW_SIZE})")
        ax.set_title(f"{ds_name} – {method}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        # Zapis wyników
        final_scores = {name: m.get() for name, m in current_metric.items()}     
        current_result = [{
            "dataset_name": ds_name,
            "method_name": method,
            "window_size": WINDOW_SIZE,
            "steps": steps,
            "mean_history": mean_history,
            "all_histories": all_histories, 
            "final_metrics": final_scores
        }]

        prog.advance(prog_task)
        np.save(os.path.join(results_dir, f"{method}.npy"), current_result)

prog.advance(prog_task)
prog.stop()