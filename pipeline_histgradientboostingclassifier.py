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

# Waarom histgradientboosting?
# Decision trees zijn een goeie manier om de gegeven features te classifiseren. Een genre hangt sterk af van verschillende features.
# Een genre bv met hoge energie, lage dansbaarheid en een recent release jaar kan eenvoudigt gemapt worden aan een decision tree.
# Het model bouwt onderliggende de verschillende "beslissingen" die leiden naar een specifiek genre op. Bij voorspellingen maken kan dan door
# de flowchart gegaan worden tot het genre bereikt wordt.
# Bij histgradientboosting worden telkens bomen opgebouwd die op elkaar itereren. Nieuwe decision trees leren telkens van de fouten uit vorige decision trees.
# Op het einde hebben we dus vervolgens een decision tree dat de som is van alle voorgaande decision trees. 

def histgradient():
    model = HistGradientBoostingClassifier(
        min_samples_leaf=5,
        random_state=42
    )
    train_model(model, "confusion_matrix_histgradientboost", HISTGRADIENT_PARAMS)
    
