import numpy as np
from aeon.datasets import load_from_tsfile
from constants import TEST_INDICES, TEST_OUT, TRAIN_INDICES, TRAIN_OUT, UCI_PATH, COMMENT
from helpers import load_indices, write_ts
from scipy.io import loadmat

TOLERANCE = 1e-4
TRAIN_SUM = -41497.005194999925
TEST_SUM = -40388.60679199994

# Load UCI MATLAB data and extract X/y arrays
uci_data = loadmat(UCI_PATH)
X_raw = uci_data["mixout"][0]
y = uci_data["consts"][0][0][4][0]

# Remove leading and trailing zeros from each case
X = np.empty(len(X_raw), dtype="object")
for i, Xi in enumerate(X_raw):
    X[i] = np.empty(3, dtype="object")
    for j, Xij in enumerate(Xi):
        idx = np.where(Xij != 0.0)
        X[i][j] = Xij[np.arange(np.min(idx), np.max(idx) + 1)]

# Number of observations of each channel by case
len_X = np.empty((len(X), 3), dtype="int")
for i, Xi in enumerate(X):
    for j, Xij in enumerate(Xi):
        len_X[i, j] = len(Xij)

# Index of cases with channels of unequal length (there should be 39)
equal_len = np.array([(len_Xi == len_Xi[0]).all() for len_Xi in len_X])
unequal_idx = np.where(np.arange(len(X)) * ~equal_len)[0]

# Pad shorter channel(s) with zeros to equalise channel lengths
for i in unequal_idx:
    lens = len_X[i]
    max_len = max(lens)
    min_idxs = np.where(lens != max_len)[0]
    for j in min_idxs:
        pad_to_max = np.zeros((max_len - lens[j], 1))
        X[i][j] = np.append(X[i][j], pad_to_max)

# Form train split
train_idx = load_indices(TRAIN_INDICES)
X_train = X[train_idx]
y_train = y[train_idx]

# Form test split
test_idx = load_indices(TEST_INDICES)
X_test = X[test_idx]
y_test = y[test_idx]

# Check splits against aeon data
X_train_sum = sum([sum([np.sum(Xij) for Xij in Xi]) for Xi in X_train])
X_test_sum = sum([sum([np.sum(Xij) for Xij in Xi]) for y in X_test])
assert X_train_sum > (TRAIN_SUM - TOLERANCE) and X_train_sum < (TRAIN_SUM + TOLERANCE)
assert X_test_sum > (TEST_SUM - TOLERANCE) and X_train_sum < (TEST_SUM + TOLERANCE)

# Write data to file
write_ts(TRAIN_OUT, X_train, y_train, COMMENT)
write_ts(TEST_OUT, X_test, y_test, COMMENT)

# Load and validate train .ts file
X_train_ts, y_train_ts = load_from_tsfile(TRAIN_OUT)
X_train_ = [np.stack(Xi) for Xi in X_train]
assert all([np.array_equal(Xi, Xi_ts) for Xi, Xi_ts in zip(X_train_, X_train_ts)])
assert np.array_equal(y_train.astype(str), y_train_ts)

# Validate and validate test .ts file
X_test_ts, y_test_ts = load_from_tsfile(TEST_OUT)
X_test_ = [np.stack(Xi) for Xi in X_test]
assert all([np.array_equal(Xi, Xi_ts) for Xi, Xi_ts in zip(X_test_, X_test_ts)])
assert np.array_equal(y_test.astype(str), y_test_ts)
