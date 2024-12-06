import pandas as pd

def filter_and_fix_csv(input_csv, output_csv, filtered_ids_path):
    df = pd.read_csv(input_csv)
    filtered_df = df[['cord_uid', 'title', 'abstract']].copy()
    for column in ['title', 'abstract']:
        filtered_df[column] = filtered_df[column].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x) if not pd.isnull(x) else None
        )
    filtered_cord_uids = filtered_df[
        filtered_df[['title', 'abstract']].isnull().any(axis=1)
    ]['cord_uid'].tolist()
    with open(filtered_ids_path, 'w') as f:
        f.writelines(f"{uid}\n" for uid in filtered_cord_uids)
    print(f"Filtered cord_uid saved to {filtered_ids_path}")
    filtered_df = filtered_df.dropna(subset=['title', 'abstract'], how='any')
    filtered_df = filtered_df.drop_duplicates(subset='cord_uid', keep='first')
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered and fixed CSV saved to {output_csv}")

input_csv = "./data/metadata.csv"
output_csv = "./data/processed_metadata.csv"
filtered_ids_path = "./data/filtered_cord_uids.txt"
filter_and_fix_csv(input_csv, output_csv, filtered_ids_path)
