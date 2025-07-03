import pandas as pd
import matplotlib.pyplot as plt

# Load CSV containing mRMR cross-validation results
df = pd.read_csv("mrmr_feature_set_scores.csv")

# Plot the performance trend
plt.figure(figsize=(8, 5))
plt.plot(df["n_features"], df["f1_macro"], marker="o", color="darkgreen")
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("Macro-F1 Score")
plt.title("mRMR Feature Subset Performance")
plt.grid(True)
plt.xticks(df["n_features"])
plt.tight_layout()

# Save the figure
plt.savefig("mrmr_subset_performance.png", dpi=300)
plt.show()
