import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based import MRMR


def run_mrmr_feature_selection(X, y, n_features=50, results_dir="results/step2_feature_selection/mrmr"):
    """Perform mRMR feature selection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or array-like
        Target labels.
    n_features : int, optional
        Number of features to select. Studies on microbiome and gene
        expression data often report stable performance using around
        50 mRMR features, so this is used as the default value.
    results_dir : str
        Directory to store the feature list and plots.

    Returns
    -------
    list
        Selected feature names ordered by importance.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Discretize features for the mutual information computation
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    X_disc = discretizer.fit_transform(X)
    y_array = pd.Series(y).values

    idx = MRMR.mrmr(X_disc, y_array, mode="index", n_selected_features=n_features)
    selected_features = X.columns[idx].tolist()

    # Save selected feature list
    list_path = os.path.join(results_dir, "mrmr_selected_features.txt")
    with open(list_path, "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    # Save ranking information
    ranking_df = pd.DataFrame({"feature": selected_features, "rank": range(1, len(selected_features)+1)})
    ranking_csv = os.path.join(results_dir, "mrmr_feature_ranking.csv")
    ranking_df.to_csv(ranking_csv, index=False)

    # Simple bar plot of top features
    plt.figure(figsize=(10, max(6, len(selected_features) * 0.25)))
    plt.barh(range(len(selected_features)), range(len(selected_features), 0, -1))
    plt.yticks(range(len(selected_features)), selected_features)
    plt.xlabel("Rank (1=best)")
    plt.title("Top mRMR Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mrmr_feature_ranking.png"), dpi=300)
    plt.close()

    return selected_features
