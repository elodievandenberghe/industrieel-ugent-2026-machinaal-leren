#!/usr/bin/env python3
from sklearn.neural_network import MLPClassifier
from framework import train_model
from scipy.stats import loguniform, randint, uniform
MLP_PARAMS = {
    "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128), (64,)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": loguniform(1e-5, 1e-1),
    "model__learning_rate_init": loguniform(1e-4, 1e-2),
    "model__batch_size": [128, 256, 512],
    "model__n_iter_no_change": [10, 20],
}
def mlp():
    model= MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                              early_stopping=True, random_state=42)
    train_model(model, "confusion_matrix_mlp", MLP_PARAMS)
    
