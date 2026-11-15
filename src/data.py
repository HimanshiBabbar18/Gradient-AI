"""Data loading, target creation, splitting and the preprocessing pipeline."""
import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

UCI_ZIP = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def _extract_csvs(zf: zipfile.ZipFile):
    """Extract student-*.csv from a zip, also looking inside nested zips."""
    for name in zf.namelist():
        base = name.split("/")[-1]
        if base in ("student-por.csv", "student-mat.csv"):
            (C.DATA_DIR / base).write_bytes(zf.read(name))
        elif name.lower().endswith(".zip"):
            _extract_csvs(zipfile.ZipFile(io.BytesIO(zf.read(name))))


def download_dataset():
    print(f"[data] downloading dataset from {UCI_ZIP}")
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(UCI_ZIP, timeout=60).read()
    _extract_csvs(zipfile.ZipFile(io.BytesIO(raw)))


def load_raw(dataset: str = "por") -> pd.DataFrame:
    """Load student-por.csv (649 rows) or student-mat.csv (395 rows)."""
    path = C.DATA_DIR / f"student-{dataset}.csv"
    if not path.exists():
        try:
            download_dataset()
        except Exception as exc:  # no internet etc.
            raise FileNotFoundError(
                f"{path} not found and automatic download failed ({exc}).\n"
                "Download the zip from https://archive.ics.uci.edu/dataset/320/student+performance "
                f"and copy student-{dataset}.csv into {C.DATA_DIR}"
            )
    df = pd.read_csv(path, sep=";")
    missing = set(C.ALL_FEATURES + [C.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected columns, missing: {missing}")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 3-class employability grade band derived from G3."""
    df = df.copy()
    df["grade_class"] = df[C.TARGET].apply(C.grade_to_class)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic quality checks. The UCI data has no missing values; we verify
    that and drop exact duplicate rows if any."""
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] rows: {n0} -> {len(df)} | missing values: {int(df.isna().sum().sum())}")
    return df


def split(df: pd.DataFrame, drop_cols=()):
    """Stratified train/test split. Returns raw (un-encoded) DataFrames."""
    feats = [c for c in C.ALL_FEATURES if c not in drop_cols]
    X = df[feats]
    y = df["grade_class"]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)


def build_preprocessor(columns) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical ones.
    Works on whichever subset of columns is passed (needed after feature selection)."""
    num = [c for c in columns if c in C.NUMERIC]
    bin_ = [c for c in columns if c in C.BINARY]
    nom = [c for c in columns if c in C.NOMINAL]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), bin_),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nom),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
