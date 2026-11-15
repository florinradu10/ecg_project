import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: int = 360,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Aplică un filtru trece-bandă Butterworth fără defazaj (filtfilt)
    pentru eliminarea derivei liniei de bază (<0.5 Hz) și a zgomotului (>40 Hz).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Standardizează semnalul (Z-score: medie 0, deviație standard 1)
    util pentru compararea între pacienți diferiți și pentru modelul ML.
    """
    std = np.std(signal)
    if std == 0:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def preprocess_ecg(signal: np.ndarray, fs: int = 360, normalize: bool = False) -> np.ndarray:
    """Pipeline complet de preprocesare: filtrare trece-bandă + normalizare opțională."""
    cleaned = bandpass_filter(signal, fs=fs)
    if normalize:
        cleaned = normalize_signal(cleaned)
    return cleaned
