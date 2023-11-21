import numpy as np
import pandas as pd


def aeon_0(X: np.array, y: np.array) -> pd.DataFrame:
    """
    Extract first observation of each case from aeon data.

    Args:
        X: Sequence data as nested NumPy array.
        y: Class labels as NumPy array.

    Returns:
        DataFrame with first x, y, z observation and class label for each case.
    """
    out = pd.DataFrame({"x": [], "y": [], "z": [], "label": []})
    for i, case in X.iterrows():
        out.loc[i] = [
            case.iloc[0][0],
            case.iloc[1][0],
            case.iloc[2][0],
            float(y[i]),
        ]
    return out


def find_case(case: pd.DataFrame, target: pd.DataFrame) -> int:
    """
    Find index of a aeon case in target (UCI) data.

    Args:
        case: Case to find in UCI data.
        target: DataFrame with UCI data.

    Returns:
        Target index for closest case.
    """
    candidates = target[target.label == case.label].iloc[:, 0:3]
    x = np.abs((candidates - case[0:3])) < 1e-6
    y = x.sum(axis=1) == 3
    assert np.sum(y) == 1
    return y[y].index.item()


def save_split(filename: str, indices: list[int]) -> None:
    """
    Save list of indices to file.

    Args:
        filename: Save data to this file.
        indices: List of integer UCI indices for train/test splits.
    """
    with open(filename, "w") as file:
        file.write("\n".join(str(i) for i in indices))


def load_indices(filename: str) -> list[int]:
    """
    Load train/test indices from file.

    Args:
        filename: Path to file with indices (one per line).

    Returns:
        List of indices.
    """
    with open(filename) as file:
        return [int(line.rstrip()) for line in file]


def write_ts(filename: str, X: np.array, y: np.array, comment: str) -> None:
    """
    Write CharacterTrajectories data to .ts file. Based on
    https://github.com/aeon-toolkit/aeon/blob/6052a179eb2b5289c99440ad3e88218bbc8b2ea1/aeon/datasets/_data_writers.py#L10.

    Args:
        filename: Save data to this file.
        X: Sequence data.
        y: Class labels.
        comment: Comment to add to .ts file.

    Returns:
        None
    """
    n_cases = len(X)
    n_channels = 3
    class_labels = np.arange(1, 21)
    class_labels = " ".join(str(label) for label in class_labels)
    with open(filename, "w") as file:
        file.write(comment)
        file.write("@problemname CharacterTrajectories\n")
        file.write("@timestamps false\n")
        file.write("@missing false\n")
        file.write("@univariate false\n")
        file.write("@dimension 3\n")
        file.write("@equallength false\n")
        file.write(f"@classlabel true {class_labels}\n")
        file.write("@data\n")
        for i in range(n_cases):
            for j in range(n_channels):
                series = ",".join([str(num) for num in X[i][j]])
                file.write(str(series))
                file.write(":")
            file.write(str(y[i]))
            file.write("\n")
