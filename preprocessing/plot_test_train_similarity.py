from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.sparse import coo_array

import vnn.variable_nearest_neighbor as vnn_lib

datasets = pd.read_csv(Path(__file__).parents[1] / "data" / "dataset_config.csv")


def plot_train_test_similarity(
    dataset: str,
    display_name: str,
    out_file: Path,
    n_neighbors: int = 5,
):
    # Load dataset
    train_csv = f"C:/Users/zsmith/Documents/GitHub/AutoML/data/preprocessed/{dataset}/morgan_fp.train.csv"
    test_csv = f"C:/Users/zsmith/Documents/GitHub/AutoML/data/preprocessed/{dataset}/morgan_fp.test.csv"
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Separate out the fingerprints
    train_fp = train[(col for col in train.columns if "FEATURE_" in col)].to_numpy()
    test_fp = test[(col for col in test.columns if "FEATURE_" in col)].to_numpy()

    # Get the matrix of similarities between test and train fingerprints
    tanimoto_mat = vnn_lib.tanimoto_matrix(coo_array(test_fp), coo_array(train_fp))

    # Get the highest n distances and average
    tanimoto_mat.sort()
    test_train_nn_dist = tanimoto_mat[:, -n_neighbors:]
    mean_nn_dist = test_train_nn_dist.mean(axis=1)

    # Plot
    fig = plt.figure(figsize=(10, 6))
    plt.hist(mean_nn_dist, range=(0, 1), bins=50)
    ax = fig.axes[0]
    y_lower, y_upper = ax.get_ybound()
    plt.vlines(
        [mean_nn_dist.mean()], y_lower, y_upper, linestyles="dashed", colors="black"
    )
    ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    ax.grid(color="#55555530")
    ax.set_ybound(upper=y_upper)
    ax.set_xbound(lower=0, upper=1)
    ax.set_title(display_name)
    ax.set_xlabel("Mean Tanimoto Similarity with 5 Nearest Neighbors")
    ax.set_ylabel("Count")
    ax.legend(
        handles=[Line2D([], [], color="black", linestyle="dashed")],
        labels=[f"Mean = {mean_nn_dist.mean():.2f}"],
    )

    # Save plot
    fig.savefig(out_file)


def main():
    directory = Path("visualization/out/test_train_similarity")
    directory.mkdir(parents=True, exist_ok=True)

    for _, dataset, display_name in datasets[["DATASET", "DISPLAY_NAME"]].itertuples():
        plot_train_test_similarity(
            dataset=dataset,
            display_name=display_name,
            out_file=directory / f"{dataset}.png"
        )

if __name__ == "__main__":
    main()