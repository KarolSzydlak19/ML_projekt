import numpy as np
from river import cluster, datasets, stream, utils, metrics
from estream import EStream
from stream_generator import StreamGenerator

SEED = 42

SPLITS_DIR = "_splits"
RESULTS_DIR = "_results"
REPORTS_DIR = "_reports"
PLOT_DIR = "_plots"

# Clusteam
N_MACRO_CLUSTERS = 3
MAX_MICRO_CLUSTERS = 100
TIME_WINDOW = 200
MICRO_CLUSTER_R_FACTOR = 1.0
TIME_GAP = 50

# Denstream
DECAYING_FACTOR = 0.1
BETA = 0.75
MU = 4
EPSILON = 0.05
N_SAMPLES_INIT = 1000
STREAM_SPEED = 100

ESTREAM_ALPHA = 20
ESTREAM_EPSILON = 0.8
ESTREAM_LAMBDA = 0.001
ESTREAM_MIN_WEIGHT = 0.001
ESTREAM_MIN_ACTIVE = 5.0

stream_generator = StreamGenerator()
N_SAMPLES = 5000
WARMUP=500
WINDOW_SIZE = 500

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
    ),

    "estream": EStream(
        n_macro_clusters=N_MACRO_CLUSTERS,
        epsilon=ESTREAM_EPSILON,
        fading_factor=ESTREAM_LAMBDA,
        alpha=ESTREAM_ALPHA,
        min_weight=ESTREAM_MIN_WEIGHT,
        min_active=ESTREAM_MIN_ACTIVE
    )
}

INSECT_VARIANT = 'gradual_balanced'

DATASETS = {
    #"HTTP": datasets.HTTP(),
    #"CreditCard": datasets.CreditCard(),
    #"ImageSegments": datasets.ImageSegments()
    #'None': lambda: [
        #stream_generator.get_stream('none').take(N_SAMPLES),
        #stream_generator.get_stream('none', seed=12).take(N_SAMPLES),
        #stream_generator.get_stream('none', seed=65321).take(N_SAMPLES),
        #stream_generator.get_stream('none', seed=1221231).take(N_SAMPLES),
        #stream_generator.get_stream('none', seed=102).take(N_SAMPLES)
    #]
    'Sudden': lambda: [
        stream_generator.get_stream('sudden').take(N_SAMPLES),
        stream_generator.get_stream('sudden', seed=12).take(N_SAMPLES),
        stream_generator.get_stream('sudden', seed=65321).take(N_SAMPLES),
        stream_generator.get_stream('sudden', seed=1221231).take(N_SAMPLES),
        stream_generator.get_stream('sudden', seed=102).take(N_SAMPLES)
    ]
    #'Gradual': lambda: [
    #    stream_generator.get_stream('gradual', n_samples=N_SAMPLES),
    #    stream_generator.get_stream('gradual', seed=12, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('gradual', seed=65321, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('gradual', seed=1221231, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('gradual', seed=102, n_samples=N_SAMPLES)
    #],
    #'Incremental': lambda: [
    #    stream_generator.get_stream('incremental', n_samples=N_SAMPLES),
    #    stream_generator.get_stream('incremental', seed=12, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('incremental', seed=65321, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('incremental', seed=1221231, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('incremental', seed=102, n_samples=N_SAMPLES)
    #],
    #'Recurring': lambda: [
    #    stream_generator.get_stream('recurring', n_samples=N_SAMPLES),
    #    stream_generator.get_stream('recurring', seed=12, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('recurring', seed=65321, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('recurring', seed=1221231, n_samples=N_SAMPLES),
    #    stream_generator.get_stream('recurring', seed=102, n_samples=N_SAMPLES)
    #]
}

METRICS = {
    "AdjustedRand":metrics.AdjustedRand(),
    "NormalizedMutualInfo": metrics.NormalizedMutualInfo(),
    "Homogeneity": metrics.Homogeneity()
}