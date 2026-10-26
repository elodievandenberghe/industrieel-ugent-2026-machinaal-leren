#!/usr/bin/env python3
from framework import train_model
from sklearn.ensemble import HistGradientBoostingClassifier 
from scipy.stats import loguniform, randint, uniform
HISTGRADIENT_PARAMS = {
    "model__learning_rate": loguniform(1e-2, 3e-1),
    "model__max_iter": randint(200, 800),
    "model__max_leaf_nodes": randint(15, 127),
    "model__max_depth": [None, 6, 10, 14],
    "model__min_samples_leaf": randint(5, 100),
    "model__l2_regularization": loguniform(1e-3, 10),
    "model__max_bins": [127, 255],
}
def histgradient():
    model = HistGradientBoostingClassifier(
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    train_model(model, "confusion_matrix_histgradientboost", HISTGRADIENT_PARAMS)
    
