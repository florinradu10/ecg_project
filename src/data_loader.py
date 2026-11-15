from pathlib import Path
import numpy as np
import pandas as pd

FS = 360  # Frecvența de eșantionare standard MIT-BIH (Hz)


def load_ecg_signal(record_id: str | int, folder: str = "data/demo", lead: str = "MLII") -> tuple[np.ndarray, np.ndarray]:
    """Încarcă semnalul ECG de pe derivația specificată și generează vectorul de timp (în secunde)."""
    file_path = Path(folder) / f"{record_id}_ekg.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Fișierul ECG nu a fost găsit: {file_path}")

    df = pd.read_csv(file_path)
    if lead not in df.columns:
        raise ValueError(
            f"Derivația '{lead}' nu există în {file_path}. Coloane disponibile: {list(df.columns)}")

    signal = df[lead].interpolate().bfill().ffill().to_numpy(dtype=float)
    time_axis = np.arange(len(signal)) / FS
    return time_axis, signal


def load_annotations(record_id: str | int, folder: str = "data/demo") -> pd.DataFrame:
    """Încarcă adnotările de referință (indexul eșantionului și simbolul bătăii)."""
    file_path = Path(folder) / f"{record_id}_annotations_1.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fișierul de adnotări nu a fost găsit: {file_path}")

    return pd.read_csv(file_path)


def load_patient_info(record_id: str | int, info_path: str = "data/pacienti_info.csv") -> dict:
    """Returnează datele clinice (vârstă, tensiune, SpO2) asociate unui record_id."""
    file_path = Path(info_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fișierul de metadate nu a fost găsit: {file_path}")

    df = pd.read_csv(file_path)
    row = df[df["record_id"].astype(str) == str(record_id)]
    if row.empty:
        return {}

    return row.iloc[0].to_dict()
