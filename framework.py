#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
DATA_PATH = "Spotify_tracks_final.csv"
TARGET = "genre"
AUDIO_FEATURES = [
    "loudness", "acousticness", "energy", "instrumentalness",
    "danceability", "valence", "release_year", "duration_ms",
    "tempo", "liveness", "speechiness", "popularity", "mode", "key",
]

GENRE_MAPPING = {
    "alt-rock": "Rock",
    "alternative": "Rock",
    "hard-rock": "Rock",
    "rock": "Rock",
    "punk": "Rock",
    "punk-rock": "Rock",
    "rock-n-roll": "Rock",
    "rockabilly": "Rock",
    "groove": "Rock",
    "grunge": "Rock",
    "goth": "Rock",
    "hardcore": "Rock",
    "psych-rock": "Rock",
    "power-pop": "Rock",
    "indie-rock": "Rock",
    "emo": "Rock",
    "j-rock": "Rock",
    "guitar": "Rock",

    "metal": "Metal",
    "metalcore": "Metal",
    "death-metal": "Metal",
    "black-metal": "Metal",
    "grindcore": "Metal",

    "pop": "Pop",
    "indie-pop": "Pop",
    "synth-pop": "Pop",
    "k-pop": "Pop",
    "j-pop": "Pop",
    "j-idol": "Pop",
    "cantopop": "Pop",
    "mandopop": "Pop",
    "pop-film": "Pop",
    
    "dance": "Electronic",
    "edm": "Electronic",
    "electronic": "Electronic",
    "house": "Electronic",
    "deep-house": "Electronic",
    "techno": "Electronic",
    "trance": "Electronic",
    "dubstep": "Electronic",
    "drum-and-bass": "Electronic",
    "breakbeat": "Electronic",
    "chicago-house": "Electronic",
    "detroit-techno": "Electronic",
    "minimal-techno": "Electronic",
    "progressive-house": "Electronic",
    "club": "Electronic",
    "dub": "Electronic",
    "idm": "Electronic",
    "hardstyle": "Electronic",
    
    "hip-hop": "Hip-Hop/R&B",
    "r-n-b": "Hip-Hop/R&B",
    "reggae": "Hip-Hop/R&B",
    "reggaeton": "Hip-Hop/R&B",
    "dancehall": "Hip-Hop/R&B",
    
    "jazz": "Jazz/Blues",
    "blues": "Jazz/Blues",
    
    "classical": "Classical/Opera",
    "opera": "Classical/Opera",
    "piano": "Classical/Opera",
    "romance": "Classical/Opera",
    "show-tunes": "Classical/Opera",
    
    "country": "Country/Folk",
    "folk": "Country/Folk",
    "bluegrass": "Country/Folk",
    "honky-tonk": "Country/Folk",
    "singer-songwriter": "Country/Folk",
    "songwriter": "Country/Folk",
    
    "latin": "Latin",
    "latino": "Latin",
    "salsa": "Latin",
    "samba": "Latin",
    "brazil": "Latin",
    "forro": "Latin",
    "mpb": "Latin",
    "pagode": "Latin",
    "sertanejo": "Latin",

    "funk": "Funk/Disco",
    "disco": "Funk/Disco",
}
def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False
    )
    fig.savefig(f"./confusion_matrices/{out_path}.png")  
    plt.close(fig)

def evaluate(model, X_test, y_test, out_path):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        f.write(classification_report(y_test, y_pred, zero_division=0))
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    print("\nPermutation importances (test set):")
    print(importances.sort_values(ascending=False))
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write("\nPermutation importances (test set):")
        f.write("{importances.sort_values(ascending=False)}")
    plot_confusion_matrix(y_test, y_pred, model.classes_, out_path)


def train_model(model, out_path, param_field):
    df = (
        pd.read_csv(DATA_PATH, index_col=0)
        .drop_duplicates(subset=["track_id"], keep="first")
        .assign(**{TARGET: lambda d: d[TARGET].map(GENRE_MAPPING)})
        .dropna(subset=[TARGET])
        .copy()
    )

    df["release_year"] = pd.to_datetime(
        df["album_release_date"], format="mixed", errors="coerce"
    ).dt.year

    y = df[TARGET]
    X = df[AUDIO_FEATURES]

    preprocess = ColumnTransformer(
        [("key_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["key"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.176,
        stratify=y_train_val,
        random_state=42
    )
    print(f"Dataset: {len(X)} samples ({len(X_train)} train, {len(X_test)} test, {len(X_val)} val))\n")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"Dataset: {len(X)} samples ({len(X_train)} train, {len(X_test)} test, {len(X_val)} val))\n")
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("scale", StandardScaler()),
        ("model", model),
    ])
    gs = RandomizedSearchCV(estimator=pipeline, cv=5, param_distributions=param_field, scoring="accuracy", n_jobs=-1, random_state=42, refit=True)
    idx = X_train.sample(n=100000, random_state=42).index
    gs.fit(X_train.loc[idx], y_train.loc[idx])
    print(gs.best_params_, gs.best_score_)
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{gs.best_params_, gs.best_score_}")
    best_model = gs.best_estimator_       
    validation_accuracy = best_model.score(X_val, y_val)
    print(f"Vallidation accuracy: {validation_accuracy}")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"Vallidation accuracy: {validation_accuracy}")
    best_model.fit(X_train, y_train)
    evaluate(best_model, X_test, y_test, out_path)
