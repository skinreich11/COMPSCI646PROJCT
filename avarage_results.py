import pandas as pd

def compute_filtered_averages(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df.drop(columns=['claim', 'topic_id'])
    df = df.loc[:, ~df.columns.str.contains('diversity_level@')]
    column_averages = df.mean().to_frame().T 
    column_averages.to_csv(output_csv, index=False)
    print(f"Averages saved to {output_csv}")

csv_file = "diversity_results.csv"
output_file = "avarage_results.csv"
averages = compute_filtered_averages(csv_file,output_file)