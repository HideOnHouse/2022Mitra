import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from utils import *


def main():
    dataset_train = GroupDataset(train=True)
    dataset_valid = GroupDataset(train=False)
    x_train, y_train = dataset_train.get_xy()
    x_valid, y_valid = dataset_valid.get_xy()
    with open('exp.txt', 'w') as f:
        feature_names = dataset_train.get_feature_names()
        n = len(feature_names)
        for num_feature in trange(n // 2, n):
            for sel_feature in combinations(range(len(feature_names)), num_feature):
                f.write(
                    f"selected feature: {[feature_names[idx] for idx in sel_feature]}" + "\n")
                x_t_partial = x_train[:, sel_feature]
                x_v_partial = x_valid[:, sel_feature]
                model = RandomForestClassifier(
                    n_jobs=-1, random_state=GLOBAL_SEED)
                model.fit(x_t_partial, y_train)
                y_pred = model.predict(x_v_partial)
                f1 = f1_score(y_valid, y_pred)
                f.write(str(f1) + '\n\n')
                print(f1)


if __name__ == "__main__":
    main()
