import requests
import json
import pandas as pd

# Define API endpoint
url = 'https://gmrepo.humangut.info/api/get_all_phenotypes'

# Make POST request to GMrepo
response = requests.post(url, data={})
phenotype_data = response.json().get('phenotypes')

# Convert to DataFrame
phenotypes_df = pd.DataFrame(phenotype_data)

# Print column names and sample rows
print("Columns:", list(phenotypes_df.columns))
print("\nSample rows:\n", phenotypes_df.head())

# Save to CSV
phenotypes_df.to_csv("all_phenotypes_from_api.csv", index=False)
print("\nSaved to 'all_phenotypes_from_api.csv'")
