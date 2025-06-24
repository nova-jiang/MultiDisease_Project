# ======================================================================== #
# data.preprocessing.py                                                    #
#                                                                          #
# Full data preprocessing pipeline after disease-class selection step,     #
# before feature selection step.                                           #
# ======================================================================== #

import pandas as pd
import requests
import json
import ast
import time
from tqdm import tqdm
from collections import defaultdict

# step 1: fetch all run IDs by MeSH disease

def fetch_all_run_ids():
    """
    Fetch all GMRepo run IDs associated with selected MeSH disease IDs.
    Saves run_id, MeSH ID, and disease category to CSV.
    """
    mesh_file = "categorised_diseases_list.tsv"
    df = pd.read_csv(mesh_file, sep="\t")
    mesh_to_category = dict(zip(df["mesh_id"], df["category"]))

    def fetch_for_mesh_id(mesh_id):
        count_url = 'https://gmrepo.humangut.info/api/countAssociatedRunsByPhenotypeMeshID'
        try:
            count_resp = requests.post(count_url, data=json.dumps({'mesh_id': mesh_id}))
            count_resp.raise_for_status()
            result = count_resp.json()
            if isinstance(result, list) and 'nr_assoc_runs' in result[0]:
                total_runs = result[0]['nr_assoc_runs']
            else:
                return []
        except:
            return []

        all_runs = []
        for skip in range(0, total_runs, 100):
            query = {"mesh_id": mesh_id, "skip": skip, "limit": 100}
            try:
                run_resp = requests.post('https://gmrepo.humangut.info/api/getAssociatedRunsByPhenotypeMeshIDLimit', data=json.dumps(query))
                run_resp.raise_for_status()
                for run_id in run_resp.json():
                    all_runs.append({
                        "run_id": run_id,
                        "mesh_id": mesh_id,
                        "category": mesh_to_category.get(mesh_id, "uncategorised")
                    })
            except:
                continue
            time.sleep(0.5)
        return all_runs

    print("Fetching run IDs...")
    all_results = []
    for mesh_id in tqdm(df["mesh_id"].unique(), desc="Fetching MeSH runs"):
        all_results.extend(fetch_for_mesh_id(mesh_id))

    df_runs = pd.DataFrame(all_results)
    df_runs.to_csv("gmrepo_all_run_ids_by_category.csv", index=False)
    print("Saved run IDs to gmrepo_all_run_ids_by_category.csv")


# step 2: fetch and expand sample metadata

def expand_run_metadata():
    """
    Expands run-level metadata from nested JSON fields into a flat DataFrame.
    Outputs full metadata including mesh_id and category.
    """
    df = pd.read_csv("gmrepo_all_run_ids_by_category.csv")

    def parse_json(row):
        if isinstance(row, str) and row.startswith("{"):
            try:
                return ast.literal_eval(row)
            except:
                return {}
        return {}

    meta_df = df["run_id"].apply(parse_json).apply(pd.Series)
    meta_df["mesh_id"] = df["mesh_id"]
    meta_df["category"] = df["category"]
    meta_df.to_csv("gmrepo_all_metadata_expanded.csv", index=False)
    print("Saved full metadata to gmrepo_all_metadata_expanded.csv")


# step 3: balance and clean metadata

def balance_metadata():
    """
    Filters and balances metadata by category and sequencing platform.
    Keeps max 240 samples per category, prioritising Illumina models.
    Drops irrelevant or sparse metadata columns.
    """
    df = pd.read_csv("gmrepo_all_metadata_expanded.csv")
    df = df[df["category"] != "Neurological"] # One neurological disease came through, deleting it here

    preferred_models = [
        "Illumina NovaSeq", "Illumina HiSeq", "Illumina NextSeq",
        "Illumina MiSeq", "Illumina Genome Analyzer"
    ]

    def instrument_rank(model):
        for i, preferred in enumerate(preferred_models):
            if isinstance(model, str) and preferred.lower() in model.lower():
                return i
        return len(preferred_models)

    df["instrument_rank"] = df["instrument_model"].apply(instrument_rank)

    balanced_df = (
        df.sort_values(["category", "instrument_rank", "nr_reads_sequenced"], ascending=[True, True, False])
          .groupby("category")
          .head(240)
          .reset_index(drop=True)
    )

    drop_cols = ["our_project_id", "more", "more_info", "accession_id", "QCStatus", "longitude", "latitude"]
    balanced_df = balanced_df.drop(columns=[col for col in drop_cols if col in balanced_df.columns])
    balanced_df.to_csv("gmrepo_balanced_metadata.csv", index=False)
    print("Saved balanced metadata to gmrepo_balanced_metadata.csv")


# step 4: build species abundance matrix

def build_abundance_matrix():
    """
    Queries GMRepo for species-level relative abundance data per run.
    Builds a full abundance matrix with species as columns and samples as rows.
    """
    df = pd.read_csv("gmrepo_balanced_metadata.csv")
    abundance_matrix = defaultdict(dict)
    species_set = set()

    def fetch_species_abundance(run_id):
        url = "https://gmrepo.humangut.info/api/getFullTaxonomicProfileByRunID"
        try:
            response = requests.post(url, data=json.dumps({"run_id": run_id}))
            response.raise_for_status()
            data = response.json()
            return {
                sp["scientific_name"]: sp["relative_abundance"]
                for sp in data.get("species", [])
                if sp.get("taxon_rank_level") == "species"
            }
        except:
            return {}

    print("Fetching species abundance...")
    for run_id in tqdm(df["run_id"]):
        abundances = fetch_species_abundance(run_id)
        for sp, val in abundances.items():
            abundance_matrix[run_id][sp] = val
            species_set.add(sp)

    species_list = sorted(species_set)
    rows = []
    for run_id in df["run_id"]:
        row = [abundance_matrix[run_id].get(sp, 0.0) for sp in species_list]
        rows.append(row)

    abundance_df = pd.DataFrame(rows, columns=species_list, index=df["run_id"])
    abundance_df = abundance_df.clip(lower=0)
    abundance_df.to_csv("species_abundance_matrix.csv")
    print(f"Saved abundance matrix with {len(abundance_df)} samples to species_abundance_matrix.csv")


# step 5: merge metadata with abundance matrix, and clean

def combine_and_clean_final_dataset():
    metadata_df = pd.read_csv("gmrepo_balanced_metadata.csv")
    abundance_df = pd.read_csv("species_abundance_matrix.csv", index_col=0)

    merged_df = metadata_df.merge(abundance_df, left_on="run_id", right_index=True)
    merged_df.to_csv("gmrepo_final_combined_dataset.csv", index=False)
    print(f"Combined dataset saved to gmrepo_final_combined_dataset.csv")

    # dropping columns with mostly empty cells
    cols_to_drop = [
        "is_disease_stage_available", "disease_stage", "collection_date", "diet",
        "Recent.Antibiotics.Use", "antibiotics_used", "Antibiotics.Dose", "Days.Without.Antibiotics.Use"
    ]
    merged_df = merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns])
    merged_df.to_csv("gmrepo_cleaned_dataset.csv", index=False)
    print(f"Final cleaned dataset saved to gmrepo_cleaned_dataset.csv with shape {merged_df.shape}")


# function to run full preprocessing pipeline

def run_full_pipeline():
    """
    Runs the complete preprocessing pipeline in sequence.
    Produces final cleaned dataset for ML pipeline input.
    """
    fetch_all_run_ids()
    expand_run_metadata()
    balance_metadata()
    build_abundance_matrix()
    combine_and_clean_final_dataset()

# execute
if __name__ == "__main__":
    run_full_pipeline()
