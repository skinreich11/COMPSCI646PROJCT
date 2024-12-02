import pandas as pd

def compute_filtered_averages(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Exclude rows where 'ndcg@10' is < 0.05
    filtered_df = df[df['ndcg@10'] >= 0.05]
    
    # Drop the first two columns ('claim', 'topic_id') and columns with 'diversity@' in their names
    filtered_df = filtered_df.drop(columns=['claim', 'topic_id'])
    filtered_df = filtered_df.loc[:, ~filtered_df.columns.str.contains('diversity_level@')]
    
    # Compute averages for the remaining numeric columns
    column_averages = filtered_df.mean().to_frame().T  # Convert to DataFrame for single-row output
    
    # Save the results to a new CSV file
    column_averages.to_csv(output_csv, index=False)
    print(f"Averages saved to {output_csv}")

csv_file = "diversity_results.csv"  # Replace with your CSV file path
output_file = "avarage_results.csv"
averages = compute_filtered_averages(csv_file,output_file)