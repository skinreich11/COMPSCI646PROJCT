import pandas as pd

def filter_and_fix_csv(input_csv, output_csv, filtered_ids_path):
    # Load the input CSV
    df = pd.read_csv(input_csv)

    # Retain only the columns 'cord_uid', 'title', and 'abstract'
    filtered_df = df[['cord_uid', 'title', 'abstract']].copy()

    # Ensure 'title' and 'abstract' are treated as strings
    for column in ['title', 'abstract']:
        filtered_df[column] = filtered_df[column].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x) if not pd.isnull(x) else None
        )
    
    # Identify `cord_uid` of documents to be filtered
    filtered_cord_uids = filtered_df[
        filtered_df[['title', 'abstract']].isnull().any(axis=1)
    ]['cord_uid'].tolist()

    # Save filtered `cord_uid` to a text file
    with open(filtered_ids_path, 'w') as f:
        f.writelines(f"{uid}\n" for uid in filtered_cord_uids)
    print(f"Filtered cord_uid saved to {filtered_ids_path}")

    # Drop rows where both 'title' and 'abstract' are missing
    filtered_df = filtered_df.dropna(subset=['title', 'abstract'], how='any')

    # Remove duplicate `cord_uid`, keeping the first occurrence
    filtered_df = filtered_df.drop_duplicates(subset='cord_uid', keep='first')

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered and fixed CSV saved to {output_csv}")

# Example usage
input_csv = "./data/metadata.csv"  # Replace with your input file path
output_csv = "./data/processed_metadata.csv"  # Replace with your desired output file path
filtered_ids_path = "./data/filtered_cord_uids.txt"  # File to save filtered cord_uid
filter_and_fix_csv(input_csv, output_csv, filtered_ids_path)
