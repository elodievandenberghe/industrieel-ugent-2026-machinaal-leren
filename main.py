#!/usr/bin/env python3
from pipeline_histgradientboostingclassifier import histgradient
from pipeline_knnearestneighbor import knn
from pipeline_mlp import mlp

def main():
    print("=====Histgradient=====")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write("=====Histgradient=====")
    histgradient()
    print("=====KNN=====")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write("=====KNN=====")
    knn()
    print("=====MLP=====")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write("=====MLP=====")
    mlp()

if __name__ == "__main__":
    main()
