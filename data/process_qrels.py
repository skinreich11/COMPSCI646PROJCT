import pandas as pd

def convert_txt_to_csv(input_txt_path, output_csv_path, filtered_ids_path):
    # Read the filtered `cord_uid` from the file
    with open(filtered_ids_path, 'r') as f:
        filtered_cord_uids = set(line.strip() for line in f)

    # Read the text file
    data = []
    with open(input_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                topic_ip, _, cord_uid, relevance = parts
                # Skip rows with `cord_uid` in the filtered list
                if cord_uid not in filtered_cord_uids:
                    data.append([topic_ip, cord_uid, relevance])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["topic_ip", "cord_uid", "relevance"])
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Filtered CSV saved to {output_csv_path}")

# Input and output file paths
input_txt_path = "./data/qrels.txt"
output_csv_path = "./data/processed_qrels.csv"
filtered_ids_path = "./data/filtered_cord_uids.txt"  # File containing filtered cord_uid

# Convert file with filtering
convert_txt_to_csv(input_txt_path, output_csv_path, filtered_ids_path)
