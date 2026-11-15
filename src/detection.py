import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(
    filtered_signal: np.ndarray,
    fs: int = 360,
    min_distance_ms: float = 200.0,
) -> np.ndarray:
    """
    Detectează indicii vârfurilor R dintr-un semnal ECG filtrat,
    folosind derivare, ridicare la pătrat, integrare pe fereastră mobilă și rafinare locală.
    """
    # 1. Derivata semnalului: evidențiază pantele foarte abrupte ale complexului QRS
    diff_signal = np.diff(filtered_signal, prepend=filtered_signal[0])

    # 2. Ridicarea la pătrat: face toate valorile pozitive și amplifică neliniar undele R față de P și T
    squared_signal = diff_signal ** 2

    # 3. Integrare pe fereastră mobilă de ~150 ms (54 eșantioane la 360 Hz) pentru a uni energia unui QRS
    window_size = max(1, int(0.150 * fs))
    kernel = np.ones(window_size) / window_size
    integrated_signal = np.convolve(squared_signal, kernel, mode="same")

    # 4. Distanța minimă fiziologică între două bătăi (200 ms = 72 eșantioane la 360 Hz)
    min_distance_samples = int((min_distance_ms / 1000.0) * fs)

    # 5. Prag adaptiv de proeminență bazat pe percentila semnalului integrat
    prominence_threshold = np.percentile(integrated_signal, 90) * 0.4

    # Găsim vârfurile aproximative pe semnalul integrat
    approx_peaks, _ = find_peaks(
        integrated_signal,
        distance=min_distance_samples,
        prominence=prominence_threshold,
    )

    # 6. Rafinare: căutăm maximul local exact pe semnalul ECG filtrat într-o fereastră de ±75 ms
    search_radius = int(0.075 * fs)
    refined_peaks = []
    n_samples = len(filtered_signal)

    for peak in approx_peaks:
        start = max(0, peak - search_radius)
        end = min(n_samples, peak + search_radius + 1)
        exact_peak = start + int(np.argmax(filtered_signal[start:end]))
        refined_peaks.append(exact_peak)

    # Eliminăm eventualele duplicate și returnăm vectorul sortat de indici
    return np.unique(np.array(refined_peaks, dtype=int))


def evaluate_detection(
    detected_peaks: np.ndarray,
    reference_indices: np.ndarray,
    fs: int = 360,
    tolerance_ms: float = 50.0,
) -> dict:
    """
    Compară vârfurile R detectate automat cu adnotările cardiologilor (toleranță standard ±50 ms)
    și calculează True Positives (TP), False Positives (FP), False Negatives (FN), Sensibilitatea și Precizia.
    """
    tol_samples = int((tolerance_ms / 1000.0) * fs)
    matched_ref = set()
    tp = 0

    for det in detected_peaks:
        diffs = np.abs(reference_indices - det)
        if len(diffs) == 0:
            continue
        nearest_idx = int(np.argmin(diffs))
        if diffs[nearest_idx] <= tol_samples and nearest_idx not in matched_ref:
            tp += 1
            matched_ref.add(nearest_idx)

    fp = len(detected_peaks) - tp
    fn = len(reference_indices) - tp

    sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity_pct": round(sensitivity, 2),
        "precision_pct": round(precision, 2),
    }
