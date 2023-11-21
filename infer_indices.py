import numpy as np
import pandas as pd
from constants import TEST_INDICES, TRAIN_INDICES, UCI_PATH
from helpers import aeon_0, find_case, save_split
from scipy.io import loadmat
from sktime.datasets import load_from_tsfile_to_dataframe as load_ts

# Path to CharacterTrajectories data
AEON_PATH_TRAIN = "data/aeon/CharacterTrajectories_TRAIN.ts"
AEON_PATH_TEST = "data/aeon/CharacterTrajectories_TEST.ts"

# Load aeon data
X_train, y_train = load_ts(AEON_PATH_TRAIN)
X_test, y_test = load_ts(AEON_PATH_TEST)

# Load UCI MATLAB data and extract X/y arrays
uci_data = loadmat(UCI_PATH)
X_uci_raw = uci_data["mixout"][0]
y_uci = uci_data["consts"][0][0][4][0]

# Remove leading and trailing zeros from each UCI case
X_uci = np.empty(len(X_uci_raw), dtype="object")
for i, Xi in enumerate(X_uci_raw):
    X_uci[i] = np.empty(3, dtype="object")
    for j, Xij in enumerate(Xi):
        idx = np.where(Xij != 0.0)
        X_uci[i][j] = Xij[np.arange(np.min(idx), np.max(idx) + 1)]

# Form DataFrame with first observation of each case in train/test aeon data
train_0 = aeon_0(X_train, y_train)
test_0 = aeon_0(X_test, y_test)

# Do the same for UCI data
uci_0 = pd.DataFrame({"x": [], "y": [], "z": [], "label": []})
for i, case in enumerate(X_uci):
    uci_0.loc[i] = [case[0][0], case[1][0], case[2][0], y_uci[i]]

# Get index in UCI data for each train/test case in aeon data
train_idx = [find_case(case, uci_0) for _, case in train_0.iterrows()]
test_idx = [find_case(case, uci_0) for _, case in test_0.iterrows()]

# Ensure no duplicate indices within each split
assert len(set(train_idx)) == len(train_idx)
assert len(set(test_idx)) == len(test_idx)

# Ensure train and test indices are disjoint
assert set(train_idx).isdisjoint(set(test_idx))

# Ensure all UCI cases are mapped to train or test split
assert set(train_idx).union(set(test_idx)) == set(np.arange(len(X_uci)))

# Write splits to file
save_split(TRAIN_INDICES, train_idx)
save_split(TEST_INDICES, test_idx)
