import pandas as pd
import requests
import json
import time
from tqdm import tqdm

# Load categorised disease list
df = pd.read_csv("categorised_diseases_list.tsv", sep="\t")
mesh_to_category = dict(zip(df["mesh_id"], df["category"]))

def fetch_all_run_ids(mesh_id):
    print(f"\nFetching for MeSH ID: {mesh_id}")

    count_url = 'https://gmrepo.humangut.info/api/countAssociatedRunsByPhenotypeMeshID'
    try:
        count_resp = requests.post(count_url, data=json.dumps({'mesh_id': mesh_id}))
        count_resp.raise_for_status()
        result = count_resp.json()
        print(f"Response for {mesh_id}: {result}")

        # Handle list response structure
        if isinstance(result, list) and len(result) > 0 and 'nr_assoc_runs' in result[0]:
            total_runs = result[0]['nr_assoc_runs']
        else:
            print(f"Unexpected format for {mesh_id}: {result}")
            return []

    except Exception as e:
        print(f"Error fetching run count for {mesh_id}: {e}")
        return []

    all_runs = []
    limit = 100

    # Now paginate through runs
    for skip in range(0, total_runs, limit):
        query = {"mesh_id": mesh_id, "skip": skip, "limit": limit}
        run_url = 'https://gmrepo.humangut.info/api/getAssociatedRunsByPhenotypeMeshIDLimit'
        try:
            run_resp = requests.post(run_url, data=json.dumps(query))
            run_resp.raise_for_status()
            runs = run_resp.json()

            for run_id in runs:
                all_runs.append({
                    "run_id": run_id,
                    "mesh_id": mesh_id,
                    "category": mesh_to_category.get(mesh_id, "uncategorised")
                })

        except Exception as e:
            print(f"Error fetching runs for {mesh_id} [skip={skip}]: {e}")
        time.sleep(0.5)  # to avoid rate limiting

    return all_runs

# Main execution
all_results = []

print("Fetching all run IDs for each MeSH disease...\n")
for mesh_id in tqdm(df["mesh_id"].unique(), desc="Fetching MeSH runs"):
    runs = fetch_all_run_ids(mesh_id)
    all_results.extend(runs)

# Save results
df_runs = pd.DataFrame(all_results)
df_runs.to_csv("gmrepo_all_run_ids_by_category.csv", index=False)

print("\nDone! Saved to gmrepo_all_run_ids_by_category.csv")
