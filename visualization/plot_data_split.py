import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from vnn.variable_nearest_neighbor import tanimoto_distance_matrix
from scipy.sparse import coo_array

RANDOM_SEED = 42
DATASET = "ames"


def plot_nearest_distance_histogram(
    train: np.ndarray,
    test: np.ndarray,
    title: str,
    save_path: str,
):
    # Get the distance between all train compounds and all test compounds
    distance_matrix = tanimoto_distance_matrix(coo_array(train), coo_array(test))
    # Get the distance to just the nearest train compound for each test compound
    test_compound_min_distance = distance_matrix.min(axis=0)

    # Plot distance to nearest train compound as a hitogram
    plt.cla()
    sns.histplot(
        test_compound_min_distance,
        kde=False,
        stat="probability",
        binrange=(0, 1),
        bins=10,
    )
    plt.xlabel("Tanimoto distance", fontsize=16)
    plt.ylabel("Proportion of compounds", fontsize=16)
    plt.ylim(0, 0.3)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def main():
    data = pd.read_csv(f"data/preprocessed/{DATASET}/morgan_fp.all.csv")
    feature_cols = [col for col in data.columns if "FEATURE_" in col]

    # Load the split data from the preprocessing step
    train_scaffold = pd.read_csv(f"data/preprocessed/{DATASET}/morgan_fp.train.csv")
    test_scaffold = pd.read_csv(f"data/preprocessed/{DATASET}/morgan_fp.test.csv")

    # Random split the data into train and test sets
    train_random, test_random = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED
    )

    # Make plots
    plot_nearest_distance_histogram(
        train=train_random[feature_cols].to_numpy(dtype=np.int8),
        test=test_random[feature_cols].to_numpy(dtype=np.int8),
        title="Random split",
        save_path="visualization/out/data_split_distance.random.png",
    )
    plot_nearest_distance_histogram(
        train=train_scaffold[feature_cols].to_numpy(dtype=np.int8),
        test=test_scaffold[feature_cols].to_numpy(dtype=np.int8),
        title="Scaffold split",
        save_path="visualization/out/data_split_distance.scaffold.png",
    )


if __name__ == "__main__":
    main()
