
import pandas as pd
import random

def add_classifications_to_cord_uids(input_csv, classification_csv, output_csv):
    """
    Adds a classification column to the input CSV based on cord_uid mappings from the classification CSV.
    If a cord_uid is not found, assigns a random classification.
    
    Parameters:
        input_csv (str): Path to the input CSV file with `cord_uids`.
        classification_csv (str): Path to the classification CSV file with `cord_uids` and `classifications`.
        output_csv (str): Path to save the updated CSV file.

    Returns:
        None
    """
    # Load the CSVs
    input_df = pd.read_csv(input_csv)
    classification_df = pd.read_csv(classification_csv)
    
    # Convert columns to lists if stored as string representations
    input_df['sorted_cord_uids'] = input_df['sorted_cord_uids'].apply(eval)
    classification_df['sorted_cord_uids'] = classification_df['sorted_cord_uids'].apply(eval)
    classification_df['sorted_scores'] = classification_df['sorted_scores'].apply(eval)

    def match_classifications(row_uids, classifications_row):
        """
        Matches `cord_uids` to their classifications.
        Assigns a random classification if not found.
        """
        classification_map = dict(zip(classifications_row['sorted_cord_uids'], classifications_row['sorted_scores']))
        classifications = []
        notFound = []
        for uid in row_uids:
            if uid in classification_map:
                classifications.append(classification_map[uid])
            else:
                print("not found", uid)
                if notFound is []:
                    ind = row_uids.index(uid)
                else:
                    ind = row_uids.index(uid) + len(notFound)
                notFound.append(ind)
        for ind in notFound:
            if ind >= len(classifications):
                if classifications[-1] <= 0.001:
                    classifications.append(0.0)
                else:
                    classifications.append(classifications[-1] - 0.001)
            else:
                middle_value = (classifications[ind - 1] + classifications[ind + 1]) / 2
                # Insert the middle value at the given index
                classifications.insert(ind, middle_value)

        return classifications

    classifications_column = []
    for i, row in input_df.iterrows():
        row_uids = row['sorted_cord_uids']
        classifications_row = classification_df.loc[i]
        classifications = match_classifications(row_uids, classifications_row)
        classifications_column.append(classifications)

    # Add the classifications column to the input DataFrame
    input_df['sorted_scores'] = classifications_column

    # Save the updated DataFrame
    input_df.to_csv(output_csv, index=False)
    print(f"Updated file saved as '{output_csv}'.")

# Example usage
if __name__ == "__main__":
    input_csv = "./Evaluation_Metrics/mmr_reranked_result.csv"  # Replace with your input CSV file
    classification_csv = "./proposed_model/singRankedListWithClass.csv"  # Replace with your classification CSV file
    output_csv = "./Evaluation_Metrics/mmr_reranked_result.csv"  # Replace with the desired output file name
    add_classifications_to_cord_uids(input_csv, classification_csv, output_csv)
