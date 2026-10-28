#!/usr/bin/env python3
from framework import train_model
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import loguniform, randint, uniform
KNN_PARAMS = {
    "model__n_neighbors": randint(5, 100),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],                      # Manhattan vs Euclidean
    "model__algorithm": ["auto", "ball_tree", "kd_tree"],
    "model__leaf_size": randint(20, 60),
}


# Bij KNN plotten we elk punt op een grafiek en kijken hoeveel x aantal van de dichtste buren gelijk zijn aan een specifiek genre.
# Gelijkaardige genres zouden dicht bij elkaar moeten liggen op deze grafiek doordat ze gelijkaardige features hebben.
# Enkele problemen hiermee zouden mogelijks kunnen zijn is overlap tussen verschillende genres. Door genres samen te groeperen zullen mogelijks genres die
# aan de rand staan van een bepaalde groep mogelijks mislabeled worden. 
def knn():
    model = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
    train_model(model, "confusion_matrix_knn", KNN_PARAMS)
    
