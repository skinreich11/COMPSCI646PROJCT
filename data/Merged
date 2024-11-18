# Merge with metadata file
metadata_file = '/Users/wentaoma/Documents/646/Final project/data/2020-07-16/metadata.csv'  # Update with actual metadata file path
merged_output_file = '/Users/wentaoma/Documents/646/Final project/data/merged_output.csv'

# Load metadata
metadata = pd.read_csv(metadata_file, usecols=['cord_uid', 'title', 'pdf_json_files'])

# Load claims file
claims = pd.read_csv(outputFile)

# Merge the filtered claims with metadata on cord_uid
merged_df = pd.concat([claims, metadata], axis=1)

# Save the merged result
merged_df.to_csv(merged_output_file, index=False)

print(f"Merged file saved to: {merged_output_file}")
