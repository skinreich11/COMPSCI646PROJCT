
import pandas as pd
import random

def add_classifications_to_cord_uids(input_csv, classification_csv, output_csv):
    input_df = pd.read_csv(input_csv)
    classification_df = pd.read_csv(classification_csv)
    input_df['sorted_cord_uids'] = input_df['sorted_cord_uids'].apply(eval)
    classification_df['sorted_cord_uids'] = classification_df['sorted_cord_uids'].apply(eval)
    classification_df['classification'] = classification_df['classification'].apply(eval)

    def match_classifications(row_uids, classifications_row):
        classification_map = dict(zip(classifications_row['sorted_cord_uids'], classifications_row['classification']))
        classifications = []

        for uid in row_uids:
            if uid in classification_map:
                classifications.append(classification_map[uid])
            else:
                print("not found", uid)
                classifications.append(random.choice(['neutral', 'contradict', 'support']))

        return classifications

    classifications_column = []
    for i, row in input_df.iterrows():
        row_uids = row['sorted_cord_uids']
        classifications_row = classification_df.loc[i]
        classifications = match_classifications(row_uids, classifications_row)
        classifications_column.append(classifications)
    input_df['classification'] = classifications_column
    input_df.to_csv(output_csv, index=False)
    print(f"Updated file saved as '{output_csv}'.")

if __name__ == "__main__":
    input_csv = "./Evaluation_Metrics/mmr_reranked_result.csv" 
    classification_csv = "./proposed_model/singRankedListWithClass.csv" 
    output_csv = "./Evaluation_Metrics/mmr_reranked_result.csv"
    add_classifications_to_cord_uids(input_csv, classification_csv, output_csv)
