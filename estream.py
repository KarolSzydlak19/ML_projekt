import numpy as np
from river import base
from sklearn.cluster import KMeans

from fcs import FCS

class EStream(base.Clusterer):
    def __init__(self,
                 n_macro_clusters=5,    # liczba makroklastrów
                 epsilon=1.0,           # max promień
                 fading_factor=0.001,   # tempo zanikania
                 alpha=20,              # liczba binów histogramu
                 min_weight=0.001,      # próg usuwania szumu
                 min_active=3.0,        # próg aktywacji
                 cleanup_interval=100):

        self.n_macro_clusters = n_macro_clusters
        self.epsilon = epsilon
        self.fading_factor = fading_factor
        self.alpha = alpha
        self.min_weight = min_weight
        self.min_active = min_active
        self.cleanup_interval = cleanup_interval

        self.FCS_list = []
        self.time_step = 0

        self.hist_min = -4.0
        self.hist_max = 4.0

    def learn_one(self, x):
        self.time_step += 1
        x_vec = np.array(list(x.values()))

        # szukamy najbliższego FCS'a
        nearest_fcs = None
        min_dist = float('inf')

        for fcs in self.FCS_list:
            dist = fcs.distance(x_vec)
            if dist < min_dist:
                min_dist = dist
                nearest_fcs = fcs

        # aktualizujemy fcs albo tworzymy nowy jeśli nei da się dopasować punktu do istniejącego fcs
        if nearest_fcs and min_dist <= self.epsilon:
            nearest_fcs.update(x_vec, self.time_step, self.fading_factor)
        else:
            new_fcs = FCS(x_vec, self.time_step, self.alpha, self.hist_min, self.hist_max)
            self.FCS_list.append(new_fcs)

        # co cleanup_interval uporządkowujemy fcs'y czyli robimy splity/merge/zmianę statusu fcsów i usuwanie
        if self.time_step % self.cleanup_interval == 0:
            self._evolution_cycle()

        return self

    def _evolution_cycle(self):
        """
        Funkcja realizuje cykl ewolucyjny:
        Zanikanie -> Sprawdzanie statusu -> Split -> Merge.
        """

        active_fcs_list = []

        # Zanikanie, status, usuwanie
        for fcs in self.FCS_list:
            time_diff = self.time_step - fcs.last_update
            if time_diff > 0:
                fcs.fade(2 ** (-self.fading_factor * time_diff))
                fcs.last_update = self.time_step

            # Usuwanie szumu (poniżej progu min)
            if fcs.weight < self.min_weight:
                continue # usunięcie poprzez niedodanie :)

            # Aktualizacja statusu
            if fcs.weight >= self.min_active:
                fcs.active = True
            else:
                fcs.active = False

            active_fcs_list.append(fcs)

        self.FCS_list = active_fcs_list

        # Split na podstawie histogramów z fcs
        next_gen_list = []
        for fcs in self.FCS_list:
            if not fcs.active:
                next_gen_list.append(fcs)
                continue

            # Pobieramy współrzędne szczytów
            dim, val_p1, val_p2 = fcs.check_split_condition()

            # SPLIT
            if dim is not None:
                # Kopiujemy stary środek jako bazę dla wszystkich fcsów
                center1 = fcs.center.copy()
                center2 = fcs.center.copy()

                # Nadpisujemy wymiar w którym robimy split
                center1[dim] = val_p1
                center2[dim] = val_p2

                # Tworzymy dzieci
                child1 = FCS(center1, self.time_step, self.alpha, self.hist_min, self.hist_max)
                child2 = FCS(center2, self.time_step, self.alpha, self.hist_min, self.hist_max)
                child1.weight = fcs.weight / 2
                child2.weight = fcs.weight / 2
                child1.LS = center1 * child1.weight
                child2.LS = center2 * child2.weight

                initial_var = (fcs.radius * 0.5) ** 2
                child1.SS = (child1.weight * (center1 ** 2)) + (child1.weight * initial_var)
                child2.SS = (child2.weight * (center2 ** 2)) + (child2.weight * initial_var)

                # Historia - dzieci zaczynają z pustymi hisogramami
                child1.parent_ids.append(fcs.id)
                child2.parent_ids.append(fcs.id)
                fcs.child_ids.extend([child1.id, child2.id])

                next_gen_list.append(child1)
                next_gen_list.append(child2)
            else:
                next_gen_list.append(fcs)

        self.FCS_list = next_gen_list

        # Merge na podstawie histogramów
        merged_list = []
        skip_indices = set()

        # Sortujemy klastry po wadze bo chcemy żeby istotniejsze klastry wchłaniały słabsze a nie na odwrotnie
        indices = np.argsort([-f.weight for f in self.FCS_list])

        for i in range(len(indices)):
            idx1 = indices[i]
            if idx1 in skip_indices: continue

            fcs1 = self.FCS_list[idx1]
            if not fcs1.active:
                merged_list.append(fcs1)
                continue

            # tu już pracujemy tylko na aktywnych klastrach
            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                if idx2 in skip_indices: continue

                fcs2 = self.FCS_list[idx2]
                if not fcs2.active: continue

                # odległość euklidesowa
                dist = np.linalg.norm(fcs1.center - fcs2.center)

                # sprawdzamy czy klastry się na siebie nakładają
                if dist < (fcs1.radius + fcs2.radius):
                    # jeśli tak to łączymy ich statystyki
                    fcs1.weight += fcs2.weight
                    fcs1.LS += fcs2.LS
                    fcs1.SS += fcs2.SS
                    fcs1.histograms += fcs2.histograms

                    # aktualizacja historii fcsa
                    fcs1.parent_ids.extend(fcs2.parent_ids)
                    fcs1.child_ids.extend(fcs2.child_ids)

                    skip_indices.add(idx2)

            merged_list.append(fcs1)

        self.FCS_list = merged_list

        active_fcs = [f for f in self.FCS_list if f.active]

        if len(active_fcs) >= self.n_macro_clusters:
            centers = np.array([f.center for f in active_fcs])
            weights = np.array([f.weight for f in active_fcs])

            # Używamy KMeans raz na cykl
            kmeans = KMeans(n_clusters=self.n_macro_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(centers, sample_weight=weights)

            # Przypisujemy etykiety do FCS-ów
            for fcs, label in zip(active_fcs, labels):
                fcs.label = int(label)

    def predict_one(self, x):
        if not self.FCS_list:
            return 0

        x_vec = np.array(list(x.values()))

        best_fcs = None
        min_dist = float('inf')

        for fcs in self.FCS_list:
            if fcs.label is None:
                continue

            dist = fcs.distance(x_vec)
            if dist < min_dist:
                min_dist = dist
                best_fcs = fcs

        if best_fcs is None:
            return 0

        return best_fcs.label