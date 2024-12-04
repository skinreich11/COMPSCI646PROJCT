import pandas as pd

def merge_csvs_by_row(input_csv_1, input_csv_2, output_csv):
    """
    Adds the `topic_ip` column from the second CSV to the first CSV based on row alignment.
    
    Parameters:
        input_csv_1 (str): Path to the first CSV file (columns: claim, sorted_cord_uids, classification).
        input_csv_2 (str): Path to the second CSV file (columns: claim, topic_ip).
        output_csv (str): Path to save the updated CSV file.
        
    Returns:
        None
    """
    # Load the CSVs
    df1 = pd.read_csv(input_csv_1)
    df2 = pd.read_csv(input_csv_2)

    # Add the `topic_ip` column from the second CSV to the first CSV
    df1['topic_ip'] = df2['topic_ip']

    # Save the updated DataFrame
    df1.to_csv(output_csv, index=False)
    print(f"Updated file saved as '{output_csv}'.")

# Example usage
if __name__ == "__main__":
    input_csv_1 = "./proposed_model/singRankedListWithClass.csv"  # Replace with your first CSV file
    input_csv_2 = "./data/claims.csv"  # Replace with your second CSV file
    output_csv = "./proposed_model/singRankedListWithClass.csv"  # Replace with the desired output file name
    merge_csvs_by_row(input_csv_1, input_csv_2, output_csv)
