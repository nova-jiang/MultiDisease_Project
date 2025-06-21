import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def run_mrmr_feature_selection(
    X,
    y,
    n_features_list=None,
    results_dir="results/step2_feature_selection/mrmr",
):
    """Perform mRMR feature selection with optional validation to find the best
    number of features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or array-like
        Target labels.
    n_features_list : list of int, optional
        List of feature counts to evaluate. If ``None`` a single set of 50
        features is returned.
    results_dir : str
        Directory to store the feature list and plots.

    Returns
    -------
    list
        Selected feature names ordered by importance for the best performing
        feature count.
    """
    print("Starting mRMR feature selection...")

    if n_features_list is None:
        n_features_list = [100]
        print("No feature count list provided. Using default: 100")

    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory at: {results_dir}")

    # Discretize features for the mutual information computation
    print("Discretizing features for mutual information computation...")
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    X_disc = discretizer.fit_transform(X)
    y_array = pd.Series(y).values

    results_summary = []

    for k in n_features_list:
        print(f"\nSelecting top {k} features using mRMR...")
        k = min(k, X.shape[1])
        idx = MRMR.mrmr(X_disc, y_array, mode="index", n_selected_features=k)
        # ``mrmr`` may return float values and a 2D shape; ensure 1D integer array
        idx = np.asarray(idx, dtype=int).ravel()[:k]
        feats = X.columns[idx].tolist()
        print(f"Selected features: {feats[:5]}{'...' if len(feats) > 5 else ''}")

        # Simple logistic regression evaluation (5-fold stratified CV)
        print("Running logistic regression with 5-fold stratified cross-validation...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline, X[feats], y, cv=cv, scoring="f1_macro"
        )
        print(f"Average F1-macro score: {scores.mean():.4f}")

        results_summary.append({"n_features": k, "score": scores.mean(), "features": feats})

    # Identify the best performing feature set
    print("\nIdentifying best performing feature set...")
    best = max(results_summary, key=lambda x: x["score"])
    selected_features = best["features"]
    print(f"Best number of features: {len(selected_features)}")

    # Save results summary
    print("Saving results summary to CSV...")
    summary_df = pd.DataFrame(
        [{"n_features": r["n_features"], "f1_macro": r["score"]} for r in results_summary]
    )
    summary_df.to_csv(os.path.join(results_dir, "mrmr_feature_set_scores.csv"), index=False)

    # Identify the best performing feature set
    best = max(results_summary, key=lambda x: x["score"])
    selected_features = best["features"]

    # Save results summary
    summary_df = pd.DataFrame(
        [{"n_features": r["n_features"], "f1_macro": r["score"]} for r in results_summary]
    )
    summary_df.to_csv(os.path.join(results_dir, "mrmr_feature_set_scores.csv"), index=False)

    # Save selected feature list
    print("Saving selected feature list...")
    list_path = os.path.join(results_dir, "mrmr_selected_features.txt")
    with open(list_path, "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    # Save ranking information
    print("Saving feature ranking to CSV...")
    ranking_df = pd.DataFrame({"feature": selected_features, "rank": range(1, len(selected_features) + 1)})
    ranking_csv = os.path.join(results_dir, "mrmr_feature_ranking.csv")
    ranking_df.to_csv(ranking_csv, index=False)

    # Simple bar plot of top features
    print("Creating bar plot of top features...")
    plt.figure(figsize=(10, max(6, len(selected_features) * 0.25)))
    plt.barh(range(len(selected_features)), range(len(selected_features), 0, -1))
    plt.yticks(range(len(selected_features)), selected_features)
    plt.xlabel("Rank (1=best)")
    plt.title("Top mRMR Features (best set: {} features)".format(len(selected_features)))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mrmr_feature_ranking.png"), dpi=300)
    plt.close()
    print("Bar plot saved.")

    print("mRMR feature selection complete.")
    return selected_features
