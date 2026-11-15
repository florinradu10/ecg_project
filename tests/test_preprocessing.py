from src.preprocessing import preprocess_ecg
from src.data_loader import load_ecg_signal, FS
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def run_test(record_id: str = "100", duration_sec: int = 10):
    time_axis, raw_signal = load_ecg_signal(
        record_id, folder="data/demo", lead="MLII")
    filtered_signal = preprocess_ecg(raw_signal, fs=FS, normalize=False)

    n_samples = duration_sec * FS
    t_win = time_axis[:n_samples]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_win, raw_signal[:n_samples], color="#7f7f7f", linewidth=1)
    axes[0].set_title(
        f"Semnal ECG Brut - Pacient {record_id} (linie de bază deplasată + zgomot)")
    axes[0].set_ylabel("Amplitudine (mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_win, filtered_signal[:n_samples],
                 color="#1f77b4", linewidth=1.2)
    axes[1].axhline(0, color="red", linestyle="--",
                    linewidth=0.8, label="Linia de bază (0 mV)")
    axes[1].set_title("Semnal ECG Filtrat (Butterworth 0.5 - 40 Hz, filtfilt)")
    axes[1].set_xlabel("Timp (secunde)")
    axes[1].set_ylabel("Amplitudine (mV)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test("100")
