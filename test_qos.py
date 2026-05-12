import joblib
import numpy as np

def indaga_modello_qos():
    percorso_modello = "coverage_prediction/modello_qos_v3.pkl"
    
    print(f"Caricamento del modello da {percorso_modello} in corso...")
    print("Nota: essendo 3.5 GB, l'operazione in RAM potrebbe richiedere diverse decine di secondi.")
    
    # 1. Carica il modello in RAM
    modello = joblib.load(percorso_modello)
    print("Modello caricato con successo!\n")
    
    # 2. Genera un set di coordinate fittizie (es. griglia 10x2 da 0 a 2000 metri)
    # Sostituisci questi limiti se la tua mappa di Esch-Belval usa un sistema di coordinate diverso
    coordinate_test = np.random.uniform(low=0, high=2000, size=(10, 2))
    
    # 3. Esegui l'inferenza
    print("Esecuzione inferenza sui punti generati...")
    predizioni = modello.predict(coordinate_test)
    
    # 4. Analisi statistica dei risultati
    print("\n--- RISULTATI DEL MODELLO QoS ---")
    print(f"Primi 10 valori grezzi estratti: {predizioni}")
    print(f"Tipo di dato (dtype): {predizioni.dtype}")
    print(f"Valore Minimo: {np.min(predizioni):.4f}")
    print(f"Valore Massimo: {np.max(predizioni):.4f}")
    print(f"Media Stimata: {np.mean(predizioni):.4f}")

if __name__ == "__main__":
    indaga_modello_qos()