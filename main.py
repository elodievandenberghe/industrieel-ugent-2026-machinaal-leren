#!/usr/bin/env python3
from pipeline_histgradientboostingclassifier import histgradient
from pipeline_knnearestneighbor import knn
from pipeline_mlp import mlp

def main():
    print("=====Histgradient=====")
    histgradient()
    print("=====KNN=====")
    knn()
    print("=====MLP=====")
    mlp()

if __name__ == "__main__":
    main()
