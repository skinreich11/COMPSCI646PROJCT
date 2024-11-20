import pandas as pd

def convert_txt_to_csv(input_txt_path, output_csv_path):
    # Read the text file
    data = []
    with open(input_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4:
                topic_ip, _, cord_uid, relevance = parts
                data.append([topic_ip, cord_uid, relevance])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["topic_ip", "cord_uid", "relevance"])
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved to {output_csv_path}")

# Input and output file paths
input_txt_path = "./data/qrels.txt"
output_csv_path = "./data/qrels.csv"

# Convert file
convert_txt_to_csv(input_txt_path, output_csv_path)