import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
import numpy as np
from river.datasets import synth
from collections import deque

class StreamGenerator:
    def __init__(self, n_features=4, n_classes=3, n_centroids=5, seed=42, seed_offset_1=0, seed_offset_2=656, visualize_stream=False):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_centroids = n_centroids
        self.seed = seed
        self.seed_offset_1 = seed_offset_1
        self.seed_offset_2 = seed_offset_2
        self.visualize_stream = visualize_stream
        
    def _get_base_stream(self, seed_offset=0):
        """Pomocnicza funkcja tworząca podstawowy strumień RBF."""
        return synth.RandomRBF(
            seed_model=self.seed + seed_offset,
            seed_sample=self.seed + seed_offset,
            n_classes=self.n_classes,
            n_features=self.n_features,
            n_centroids=self.n_centroids,
        )

    def get_stream(self, drift_type='none', n_samples=5000, seed=0):
        """
        Główna fabryka strumieni.
        Dostępne dryfy: 'none', 'incremental', 'sudden', 'gradual', 'recurring'
        """
        print(f"--- Generowanie strumienia typu: {drift_type.upper()} ---")

        # 1. BRAK DRYFU (Stationary)
        if drift_type == 'none':
            print("Zakonczone")
            return self._get_base_stream(seed_offset=seed)

        # 2. INKREMENTALNY (Incremental - ciągła, powolna zmiana)
        elif drift_type == 'incremental':
            return synth.RandomRBFDrift(
                seed_model=self.seed,
                seed_sample=self.seed,
                n_classes=self.n_classes,
                n_features=self.n_features,
                n_centroids=self.n_centroids,
                n_drift_centroids=self.n_centroids,
                change_speed=0.002  # Prędkość zmian
            )

        # 3. NAGŁY (Sudden - gwałtowna zmiana w połowie)
        elif drift_type == 'sudden':
            stream_a = self._get_base_stream(seed_offset=self.seed_offset_1)
            stream_b = self._get_base_stream(seed_offset=self.seed_offset_2)
            
            return synth.ConceptDriftStream(
                stream=stream_a,
                drift_stream=stream_b,
                position=n_samples // 2, # Zmiana w połowie
                width=50 
            )

        # 4. STOPNIOWY (Gradual - mieszanie się konceptów)
        elif drift_type == 'gradual':
            stream_a = self._get_base_stream(seed_offset=self.seed_offset_1)
            stream_b = self._get_base_stream(seed_offset=self.seed_offset_2)
            
            return synth.ConceptDriftStream(
                stream=stream_a,
                drift_stream=stream_b,
                position=n_samples // 2,
                width=1500  # Przez 1500 próbek mamy okres przejściowy
            )

        # 5. CYKLICZNY (Recurring - powrót do starego konceptu A -> B -> A)
        elif drift_type == 'recurring':
            stream_a1 = self._get_base_stream(seed_offset=self.seed_offset_1)
            stream_b = self._get_base_stream(seed_offset=self.seed_offset_2)
            stream_a2 = stream_a1 #self._get_base_stream(seed_offset=0) # Powrót do A (ten sam seed co a1)
            
            # Najpierw zmiana A -> B w 1/3 czasu
            drift_1 = synth.ConceptDriftStream(
                stream=stream_a1,
                drift_stream=stream_b,
                position=n_samples // 3,
                width=1000 # Stopniowe przejście
            )
            
            # Potem zmiana (A->B) -> A w 2/3 czasu
            drift_2 = synth.ConceptDriftStream(
                stream=drift_1,
                drift_stream=stream_a2,
                position=(n_samples // 3) * 2,
                width=1000
            )
            return drift_2

        else:
            raise ValueError("Dostepne dryfy: none, incremental, sudden, gradual, recurring")

    def visualize(self, stream, n_steps=5000):
        """Metoda do wizualizacji przekazanego strumienia."""
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True, linestyle='--', alpha=0.6)
        scat = ax.scatter([], [], c=[], cmap='viridis', s=30, alpha=0.7, edgecolor='k')
        
        buffer_len = 200
        data_buffer = deque(maxlen=buffer_len)
        labels_buffer = deque(maxlen=buffer_len)
        
        step = 0
        try:
            for x, y in stream:
                data_buffer.append(list(x.values()))
                labels_buffer.append(y)
                step += 1
                
                if step % 20 == 0 and len(data_buffer) > 0:
                    X_plot = np.array(data_buffer)
                    # Używamy cechy 0 i 1 do wizualizacji
                    scat.set_offsets(X_plot[:, :2]) 
                    scat.set_array(np.array(labels_buffer))
                    
                    ax.set_title(f"Krok: {step} | Bufor: {len(data_buffer)}")
                    plt.pause(0.001)
                
                if step >= n_steps:
                    break
        except KeyboardInterrupt:
            print("Zatrzymano.")
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    gen = StreamGenerator(n_features=10, n_classes=3, n_centroids=10, visualize_stream=True)
    
    # Opcje: 'none', 'incremental', 'sudden', 'gradual', 'recurring'
    wybrany_typ = 'incremental' 
    
    my_stream = gen.get_stream(drift_type=wybrany_typ, n_samples=3000)
    print(type(my_stream))

    if gen.visualize_stream:
        gen.visualize(my_stream, n_steps=3000)