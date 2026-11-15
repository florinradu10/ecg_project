
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from src.detection import detect_r_peaks, evaluate_detection
from src.preprocessing import preprocess_ecg
from src.data_loader import load_ecg_signal, load_annotations, FS

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# Simbolurile MIT-BIH care reprezintă bătăi cardiace reale (ignorăm simbolurile non-bătaie precum '+' sau '~')
BEAT_SYMBOLS = {"N", "L", "R", "B", "A", "a", "J", "S",
                "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"}


def run_test(record_id: str = "100", duration_sec: int = 10):
    time_axis, raw_signal = load_ecg_signal(
        record_id, folder="data/demo", lead="MLII")
    filtered_signal = preprocess_ecg(raw_signal, fs=FS, normalize=False)

    annotations = load_annotations(record_id, folder="data/demo")
    beat_annotations = annotations[annotations["annotation_symbol"].isin(
        BEAT_SYMBOLS)]
    ref_indices = beat_annotations["index"].to_numpy()

    # Rulăm detectorul pe tot semnalul de 30 de minute
    detected_peaks = detect_r_peaks(filtered_signal, fs=FS)
    metrics = evaluate_detection(detected_peaks, ref_indices, fs=FS)

    print(f"=== Evaluare Detecție Vârfuri R - Pacient {record_id} ===")
    print(
        f"Bătăi reale (MIT-BIH): {len(ref_indices)} | Bătăi detectate: {len(detected_peaks)}")
    print(
        f"TP: {metrics['TP']} | FP (alarme false): {metrics['FP']} | FN (ratate): {metrics['FN']}")
    print(
        f"Sensibilitate (Recall): {metrics['sensitivity_pct']}% | Precizie (PPV): {metrics['precision_pct']}%")

    # Vizualizare pe primele `duration_sec` secunde
    n_samples = duration_sec * FS
    t_win = time_axis[:n_samples]
    s_win = filtered_signal[:n_samples]

    det_win = detected_peaks[detected_peaks < n_samples]
    ref_win = ref_indices[ref_indices < n_samples]

    plt.figure(figsize=(12, 4))
    plt.plot(t_win, s_win, label="Semnal ECG Filtrat",
             color="#1f77b4", linewidth=1)
    plt.scatter(ref_win / FS, filtered_signal[ref_win], color="red",
                s=60, marker="o", label="Referință MIT-BIH", zorder=3)
    plt.scatter(det_win / FS, filtered_signal[det_win], color="lime", s=25,
                marker="x", linewidths=2, label="Detectat de algoritm", zorder=4)
    plt.title(
        f"Detecție Vârfuri R - Pacient {record_id} (Sensibilitate: {metrics['sensitivity_pct']}%, Precizie: {metrics['precision_pct']}%)")
    plt.xlabel("Timp (secunde)")
    plt.ylabel("Amplitudine (mV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test("100")
