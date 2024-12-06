
import pandas as pd
import random

def add_classifications_to_cord_uids(input_csv, classification_csv, output_csv):
    input_df = pd.read_csv(input_csv)
    classification_df = pd.read_csv(classification_csv)
    input_df['sorted_cord_uids'] = input_df['sorted_cord_uids'].apply(eval)
    classification_df['sorted_cord_uids'] = classification_df['sorted_cord_uids'].apply(eval)
    classification_df['sorted_scores'] = classification_df['sorted_scores'].apply(eval)

    def match_classifications(row_uids, classifications_row):
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
                classifications.insert(ind, middle_value)

        return classifications

    classifications_column = []
    for i, row in input_df.iterrows():
        row_uids = row['sorted_cord_uids']
        classifications_row = classification_df.loc[i]
        classifications = match_classifications(row_uids, classifications_row)
        classifications_column.append(classifications)
    input_df['sorted_scores'] = classifications_column
    input_df.to_csv(output_csv, index=False)
    print(f"Updated file saved as '{output_csv}'.")

if __name__ == "__main__":
    input_csv = "./Evaluation_Metrics/mmr_reranked_result.csv"
    classification_csv = "./proposed_model/singRankedListWithClass.csv"
    output_csv = "./Evaluation_Metrics/mmr_reranked_result.csv"
    add_classifications_to_cord_uids(input_csv, classification_csv, output_csv)
