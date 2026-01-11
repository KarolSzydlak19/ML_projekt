import numpy as np
import uuid

class FCS:
    """
    Fading Cluster Structure - przechowuje statystyki, histogramy oraz historię ewolucji.
    """

    def __init__(self, x, timestamp, n_bins=20, range_min=-5.0, range_max=5.0):
        self.id = uuid.uuid4()
        self.creation_time = timestamp
        self.last_update = timestamp
        self.parent_ids = []    # to do splitu
        self.child_ids = []     # a to do merga
        self.label = None

        # active/inactive
        self.active = False

        self.weight = 1.0
        self.LS = x.copy()
        self.SS = x.copy() ** 2

        # histogramy do wykrywania splitów
        self.dim = len(x)
        self.n_bins = n_bins
        self.range_min = range_min
        self.range_max = range_max
        self.bin_width = (range_max - range_min) / n_bins

        self.histograms = np.zeros((self.dim, n_bins))
        self._update_histogram(x, 1.0)

    def update(self, x, timestamp, fading_factor):
        time_diff = timestamp - self.last_update
        if time_diff > 0:
            fade = 2 ** (-fading_factor * time_diff)
            self.fade(fade)

        # Dodanie nowej instancji
        self.weight += 1.0
        self.LS += x
        self.SS += x ** 2
        self._update_histogram(x, 1.0)

        self.last_update = timestamp

    def fade(self, factor):
        """Zmniejsza wagi statystyk i histogramów."""
        self.weight *= factor
        self.LS *= factor
        self.SS *= factor
        self.histograms *= factor

    def _update_histogram(self, x, w):
        indices = np.floor((x - self.range_min) / self.bin_width).astype(int)
        indices = np.clip(indices, 0, self.n_bins - 1)
        for d in range(self.dim):
            self.histograms[d, indices[d]] += w

    @property
    def center(self):
        return self.LS / self.weight

    @property
    def radius(self):
        if self.weight <= 1e-5: return 0.0
        mean = self.center
        mean_sq = self.SS / self.weight
        variance = np.maximum(mean_sq - (mean ** 2), 0.0)
        avg_variance = np.mean(variance)
        return np.sqrt(avg_variance) * 1.5

    def distance(self, x):
        return np.linalg.norm(self.center - x)

    def check_split_condition(self):
        """
        Sprawdza histogramy w poszukiwaniu doliny (split).
        Sprawdza pary sąsiadujących szczytów i wybiera najgłębszą dolinę.
        Zwraca: (dimension_index, val_peak1, val_peak2)
        """
        if self.weight < 5.0: return None, None, None

        best_dim = None
        center_1 = None
        center_2 = None
        best_valley_score = 0.8 # stosunek dolina/szczyt)

        for d in range(self.dim):
            hist = self.histograms[d]
            avg_h = np.mean(hist)
            if avg_h < 1e-3: continue

            # Szukamy wszystkich szczytów
            peaks = []
            for i in range(1, self.n_bins - 1):
                if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                    if hist[i] > 0.3 * avg_h:
                        peaks.append(i)

            if len(peaks) < 2:
                continue

            # Sprawdzamy każdą parę sąsiadujących szczytów
            for i in range(len(peaks) - 1):
                p1 = peaks[i]
                p2 = peaks[i + 1]

                if p2 - p1 < 2: continue

                # Analiza fragmentu histogramu między p1 a p2
                segment = hist[p1 + 1:p2]
                min_valley_val = np.min(segment)

                lower_peak_height = min(hist[p1], hist[p2]) # wysokość niższego z dwóch szczytów

                valley_ratio = min_valley_val / lower_peak_height if lower_peak_height > 0 else 1.0

                if valley_ratio < 0.5:
                    # Szukamy najgłębszej doliny
                    if valley_ratio < best_valley_score:
                        best_valley_score = valley_ratio
                        best_dim = d
                        # Obliczamy centra nowych klastrów
                        center_1 = self.range_min + (p1 + 0.5) * self.bin_width
                        center_2 = self.range_min + (p2 + 0.5) * self.bin_width

        if best_dim is not None:
            return best_dim, center_1, center_2

        return None, None, None