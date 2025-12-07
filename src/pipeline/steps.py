
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm

def _p(cfg, *k):
    cur = cfg
    for kk in k:
        cur = cur[kk]
    return cur

def _log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def init_structure(cfg: Dict[str, Any], dry_run: bool = False):
    paths = cfg["paths"]
    for key in ["data_raw", "data_interim", "data_processed", "models", "logs"]:
        p = Path(paths[key])
        if dry_run:
            _log(f"(dry-run) mkdir -p {p}")
        else:
            p.mkdir(parents=True, exist_ok=True)
            _log(f"OK: {p}")

def step_collect(cfg: Dict[str, Any], dry_run: bool = False):
    raw = Path(cfg["paths"]["data_raw"])
    _log("Zbieranie danych (stub). Tu dodasz pobieranie z Kaggle/Roboflow/SoccerNet lub kopiowanie z dysku.")
    _log(f"Docelowy katalog: {raw}")
    if not dry_run:
        raw.mkdir(parents=True, exist_ok=True)

def step_extract(cfg: Dict[str, Any], dry_run: bool = False):
    inter = Path(cfg["paths"]["data_interim"]) / "frames"
    _log("Ekstrakcja klatek z wideo (stub). Tu wywołasz swój extractor (np. OpenCV).")
    _log(f"Docelowy katalog: {inter}")
    if not dry_run:
        inter.mkdir(parents=True, exist_ok=True)

def step_candidates(cfg: Dict[str, Any], dry_run: bool = False):
    inter = Path(cfg["paths"]["data_interim"]) / "candidates"
    _log("Wyszukiwanie kandydatów (stub). Tu pójdzie MSER/thresholding/YOLO detektor cyfr.")
    _log(f"Docelowy katalog: {inter}")
    if not dry_run:
        inter.mkdir(parents=True, exist_ok=True)

def step_patches(cfg: Dict[str, Any], dry_run: bool = False):
    proc = Path(cfg["paths"]["data_processed"]) / "patches"
    _log("Wycinanie patchy (stub). Tu utworzysz podkatalogi klas 0..99 i zapiszesz patche.")
    _log(f"Docelowy katalog: {proc}")
    if not dry_run:
        for c in range(10):  # przykładowe 0..9 (na start)
            (proc / str(c)).mkdir(parents=True, exist_ok=True)

def step_train(cfg: Dict[str, Any], dry_run: bool = False):
    models = Path(cfg["paths"]["models"])
    _log("Trening HOG+SVM (stub). Tu wczytasz patche -> HOG -> SVM i zapiszesz model.")
    _log(f"Model trafi do: {models / 'hog_svm.joblib'}")
    if not dry_run:
        models.mkdir(parents=True, exist_ok=True)

def step_infer(cfg: Dict[str, Any], dry_run: bool = False):
    inter = Path(cfg["paths"]["data_interim"])
    _log("Inferencja (stub). Tu wczytasz model i generujesz predykcje/wykresy.")
    _log(f"Wyniki/ wizualizacje zapisuj do: {inter}")
