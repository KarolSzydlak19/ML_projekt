import numpy as np
import pandas as pd
import os
import glob

results_folder = os.path.join("_results", "ImageSegments")
all_files = glob.glob(os.path.join(results_folder, "*.npy"))

rows = []

METRIC_NAMES = [
    "AdjustedRand",
    "V-Measure",
    "MutualInfo",
    "Homogeneity",
    "Completeness"
]

for f in all_files:
    try:
        data = np.load(f, allow_pickle=True)

        for entry in data:
            metrics_values = entry['metrics']

            if isinstance(metrics_values, list) or isinstance(metrics_values, np.ndarray):
                metrics_dict = dict(zip(METRIC_NAMES, metrics_values))
            elif isinstance(metrics_values, dict):
                metrics_dict = metrics_values
            else:
                print(f"Nieznany typ danych w pliku {f}: {type(metrics_values)}")
                continue

            row = {
                "Method": entry['method_name'],
                **metrics_dict
            }
            rows.append(row)

    except Exception as e:
        print(f"Błąd przy odczycie pliku {f}: {e}")

if rows:
    df = pd.DataFrame(rows)
    if "Method" in df.columns:
        df.set_index("Method", inplace=True)

    print("=== WYNIKI ===")
    print(df)

else:
    print("Brak wyników do wyświetlenia.")