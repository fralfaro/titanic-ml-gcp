from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_DIR = Path("data/raw")


def load_data() -> pd.DataFrame:
    """Carga el dataset de entrenamiento Titanic."""
    path = RAW_DATA_DIR / "train.csv"
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesamiento básico del dataset Titanic."""
    df = df.copy()

    # Selección de variables
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    target = "Survived"

    df = df[features + [target]]

    # Imputación simple
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Codificación
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Separación train / test determinística."""
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
