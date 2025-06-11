# ---------------------------------- #
# MODEL BASED FEATURE SELECTION      #
# Works, but very small sample group #
# (preventing 'healthy' from         #
# overfitting)!! Ready for CV though #
# ---------------------------------- #


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# -----------------------------
# Step 1: Load & reshape abundance data
# -----------------------------
print("Loading abundance data...")
abundance_long = pd.read_csv("readed_data/species_relab.tsv", sep="\t")
abundance_long = abundance_long[abundance_long["taxon_rank_level"] == "species"]

# Pivot to wide format: sample x ncbi_taxon_id
abundance_df = abundance_long.pivot_table(
    index="loaded_uid",
    columns="ncbi_taxon_id",
    values="relative_abundance",
    fill_value=0
)
abundance_df.index = abundance_df.index.astype(str)

# -----------------------------
# Step 2: Load mapping from loaded_uid to run_id
# -----------------------------
print("Loading sample-to-run mapping...")
mapping_df = pd.read_csv("readed_data/sample_to_run_info.tsv", sep="\t", low_memory=False)
mapping_df = mapping_df[["sample_name", "run_id"]].dropna()
mapping_df["sample_name"] = mapping_df["sample_name"].astype(str)
mapping_df["run_id"] = mapping_df["run_id"].astype(str)

# Map loaded_uid (sample_name) to run_id
uid_to_run = mapping_df.set_index("sample_name")["run_id"].to_dict()
abundance_df.index = abundance_df.index.map(uid_to_run)
abundance_df = abundance_df.dropna()
abundance_df.index = abundance_df.index.astype(str)

# -----------------------------
# Step 3: Load metadata
# -----------------------------
print("Loading metadata...")
metadata_df = pd.read_csv("readed_data/sample_metadata.tsv", sep="\t")
metadata_df["run_id"] = metadata_df["run_id"].astype(str)
metadata_df["phenotype"] = metadata_df["phenotype"].str.lower()
metadata_df = metadata_df.rename(columns={"run_id": "sample_id"})
metadata_df = metadata_df.set_index("sample_id")

# -----------------------------
# Step 4: Match samples & auto-select diseases
# -----------------------------
print("Matching samples and filtering phenotypes...")
common_ids = abundance_df.index.intersection(metadata_df.index)
abundance_df = abundance_df.loc[common_ids]
metadata_df = metadata_df.loc[common_ids]

# Auto-select top 4 disease phenotypes (excluding health)
disease_only = metadata_df[metadata_df["phenotype"] != "health"]
top_diseases = disease_only["phenotype"].value_counts().head(4).index.tolist()
print("Selected disease phenotypes:", top_diseases)

target_diseases = ["health"] + top_diseases
metadata_df = metadata_df[metadata_df["phenotype"].isin(target_diseases)]
abundance_df = abundance_df.loc[metadata_df.index]

# -----------------------------
# Step 5: Balance classes
# -----------------------------
combined = pd.concat([abundance_df, metadata_df["phenotype"]], axis=1)
min_size = combined["phenotype"].value_counts().min()
balanced = pd.concat([
    resample(group, n_samples=min_size, random_state=42)
    for _, group in combined.groupby("phenotype")
])

abundance_df = balanced.drop(columns=["phenotype"])
metadata_df = balanced[["phenotype"]]

print(f"Filtered sample count: {len(abundance_df)}")
print("Phenotype distribution:\n", metadata_df["phenotype"].value_counts())

# -----------------------------
# Step 6: Normalize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(abundance_df)
y = metadata_df["phenotype"]

# -----------------------------
# Step 7: Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Step 8: Random Forest & feature selection
# -----------------------------
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

selector = SelectFromModel(rf, prefit=True, threshold="median")
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

selected_features = abundance_df.columns[selector.get_support()]
print(f"Selected features: {len(selected_features)}")
print(f"First 5 selected taxa: {selected_features[:5].tolist()}")

# -----------------------------
# Step 9: Retrain and evaluate
# -----------------------------
print("Evaluating on selected features...")
rf_final = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_final.fit(X_train_sel, y_train)
y_pred = rf_final.predict(X_test_sel)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
