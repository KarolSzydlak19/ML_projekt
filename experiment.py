import os
from river import preprocessing, compose
from rich.progress import Progress, TimeElapsedColumn
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from config import *
import pandas as pd

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
            compose.SelectType(int, float)  | preprocessing.StandardScaler()
            ) | METHODS[method].clone()
            current_metric = {
                name: metric.clone() 
                for name, metric in METRICS.items()
            }
            history = {m_name: [] for m_name in METRICS.keys()}
            steps = []
            base_dataset_list = list(base_dataset)
            for i, (x,y) in enumerate(base_dataset_list):
                if i % 1000 == 0:
                    print(f"Dataset: {ds_name} | Metoda: {method} | Krok: {i}")
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

        # tworzenie tabel do sprawka
        TABLE_INTERVAL = 500

        df_history = pd.DataFrame(mean_history)
        df_history['Step'] = steps

        indices_to_keep = list(range(0, len(steps), TABLE_INTERVAL))
        if (len(steps) - 1) not in indices_to_keep:
            indices_to_keep.append(len(steps) - 1)

        df_table = df_history.iloc[indices_to_keep].copy()

        cols = ['Step'] + [c for c in df_table.columns if c != 'Step']
        df_table = df_table[cols]

        df_table = df_table.round(4)

        tables_dir = os.path.join(RESULTS_DIR, "tables", ds_name)
        os.makedirs(tables_dir, exist_ok=True)

        csv_filename = f"{method}_summary.csv"
        csv_path = os.path.join(tables_dir, csv_filename)
        df_table.to_csv(csv_path, index=False, sep=';')
        
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
