import numpy as np
from river import datasets, preprocessing, compose
from estream import EStream
from config import *

# Konfiguracja z config.py, ale z włączonym trybem "podglądacza"
print("=== ROZPOCZYNAM DIAGNOSTYKĘ ESTREAM ===")

# Pobieramy dataset
dataset = list(datasets.ImageSegments())
print(f"Liczba instancji w zbiorze: {len(dataset)}")

# Tworzymy pipeline (ważne, żeby był StandardScaler!)
scaler = preprocessing.StandardScaler()

# Inicjalizujemy model
model = EStream(
    n_macro_clusters=N_MACRO_CLUSTERS,
    epsilon=ESTREAM_EPSILON,
    fading_factor=ESTREAM_LAMBDA,
    alpha=ESTREAM_ALPHA,
    min_weight=ESTREAM_MIN_WEIGHT,
    min_active=ESTREAM_MIN_ACTIVE,
    cleanup_interval=100  # sprawdzamy co 100 kroków
)

print(f"Parametry: Epsilon={model.epsilon}, Min_Active={model.min_active}")

# Ręczna pętla uczenia
for i, (x, y) in enumerate(dataset):

    # 1. Symulacja Pipeline'u (skalowanie)
    # River wymaga uczenia skalera w locie
    scaler.learn_one(x)
    x_scaled = scaler.transform_one(x)

    # 2. Uczenie E-Stream
    model.learn_one(x_scaled)

    # 3. Raport co 500 kroków
    if (i + 1) % 500 == 0:
        n_fcs = len(model.FCS_list)
        n_active = len([f for f in model.FCS_list if f.active])

        # Pobranie wagi największego klastra
        max_weight = 0
        if n_fcs > 0:
            max_weight = max([f.weight for f in model.FCS_list])

        print(f"\n[Krok {i + 1}]")
        print(f"  Liczba FCS: {n_fcs}")
        print(f"  Aktywne FCS: {n_active}")
        print(f"  Największa waga: {max_weight:.2f}")

        if n_fcs > 0:
            f0 = model.FCS_list[0]  # lub znajdź ten z max wagą
            # Znajdź FCS z największą wagą dla lepszego podglądu
            heavy_fcs = max(model.FCS_list, key=lambda f: f.weight)

            print(f"  Najcięższy FCS Radius: {heavy_fcs.radius:.4f}")
            # Wyświetl sumę (czy histogram w ogóle żyje) i środkowe biny
            print(f"  Histogram Suma (wymiar 0): {np.sum(heavy_fcs.histograms[0]):.2f}")
            mid = len(heavy_fcs.histograms[0]) // 2
            print(f"  Histogram Środek (wymiar 0): {heavy_fcs.histograms[0][mid - 2:mid + 3]}")

print("\n=== KONIEC ===")
# Szybki test predykcji na ostatnich 10 próbkach
print("Test predykcji (ostatnie 10 próbek):")
ids = []
for i in range(len(dataset) - 10, len(dataset)):
    x, y = dataset[i]
    x_scaled = scaler.transform_one(x)
    pred = model.predict_one(x_scaled)
    ids.append(pred)
print(f"Zwrócone ID klastrów: {ids}")