import numpy as np
from river import cluster, datasets, metrics, stream

SEED = 42

SPLITS_DIR = "_splits"
RESULTS_DIR = "_results"
REPORTS_DIR = "_reports"

# Clusteam
N_MACRO_CLUSTERS = 5
MAX_MICRO_CLUSTERS = 100
TIME_WINDOW = 1000
MICRO_CLUSTER_R_FACTOR = 2.0
TIME_GAP = 100

# Denstream
DECAYING_FACTOR = 0.25
BETA = 0.75
MU = 2
EPSILON = 0.02
N_SAMPLES_INIT = 1000
STREAM_SPEED = 100

METHODS = {
    
    "clustream": cluster.CluStream(
        n_macro_clusters=N_MACRO_CLUSTERS,
        max_micro_clusters=MAX_MICRO_CLUSTERS,
        time_window=TIME_WINDOW,
        micro_cluster_r_factor=MICRO_CLUSTER_R_FACTOR,
        time_gap=TIME_GAP,
        seed=SEED
    ),
    
    "denstream": cluster.DenStream(
        decaying_factor=DECAYING_FACTOR,
        epsilon=EPSILON,
        beta=BETA,
        mu=MU,
        n_samples_init=N_SAMPLES_INIT,
        stream_speed=STREAM_SPEED
    )
}

INSECT_VARIANT = 'gradual_balanced'

DATASETS = {
    #"HTTP": datasets.HTTP(),
    #"CreditCard": datasets.CreditCard(),
    "ImageSegments": datasets.ImageSegments()
}

METRICS = {
    "AdjustedRand": metrics.AdjustedRand(),
    "V-Measure": metrics.VBeta(),
    "MutualInfo": metrics.MutualInfo(),
    "Homogeneity": metrics.Homogeneity(),
    "Completeness": metrics.Completeness()
}