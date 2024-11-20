import pandas as pd

def filter_and_fix_csv(input_csv, output_csv):
    # Load the input CSV
    df = pd.read_csv(input_csv)

    # Retain only the columns 'cord_uid', 'title', and 'abstract'
    filtered_df = df[['cord_uid', 'title', 'abstract']].copy()

    # Ensure 'title' and 'abstract' are treated as strings
    for column in ['title', 'abstract']:
        filtered_df[column] = filtered_df[column].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x) if not pd.isnull(x) else None
        )
    
    # Drop rows where both 'title' and 'abstract' are missing or empty
    filtered_df = filtered_df.dropna(subset=['title', 'abstract'], how='all')

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered and fixed CSV saved to {output_csv}")

# Example usage
input_csv = "./data/metadata.csv"  # Replace with your input file path
output_csv = "./data/processed_metadata2.csv"  # Replace with your desired output file path
filter_and_fix_csv(input_csv, output_csv)
