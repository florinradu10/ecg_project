import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_ecg_signal, load_annotations, load_patient_info, FS


def run_test(record_id: str = "100", duration_sec: int = 10):
    time_axis, signal = load_ecg_signal(record_id, folder="data/demo", lead="MLII")
    annotations = load_annotations(record_id, folder="data/demo")
    patient = load_patient_info(record_id)

    print(f"=== Verificare Pacient {record_id} ===")
    print(f"Eșantioane totale: {len(signal)} | Durată: {time_axis[-1]:.1f} s ({time_axis[-1]/60:.1f} min)")
    print(f"Total adnotări: {len(annotations)} | Primele simboluri: {annotations['annotation_symbol'].value_counts().to_dict()}")
    print(f"Date clinice: {patient}")

    n_samples = duration_sec * FS
    t_win = time_axis[:n_samples]
    s_win = signal[:n_samples]
    ann_win = annotations[annotations["index"] < n_samples]

    plt.figure(figsize=(12, 4))
    plt.plot(t_win, s_win, label="Semnal brut MLII (mV)", color="#1f77b4", linewidth=1)
    plt.scatter(
        ann_win["index"] / FS,
        signal[ann_win["index"]],
        color="red",
        s=30,
        zorder=3,
        label="Adnotări referință (MIT-BIH)",
    )
    plt.title(f"Semnal ECG Brut - {patient.get('nume_pacient', record_id)} (primele {duration_sec}s)")
    plt.xlabel("Timp (secunde)")
    plt.ylabel("Amplitudine (mV)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test("100")