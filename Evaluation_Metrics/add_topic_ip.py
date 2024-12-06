import pandas as pd

def merge_csvs_by_row(input_csv_1, input_csv_2, output_csv):
    df1 = pd.read_csv(input_csv_1)
    df2 = pd.read_csv(input_csv_2)
    df1['topic_ip'] = df2['topic_ip']
    df1.to_csv(output_csv, index=False)
    print(f"Updated file saved as '{output_csv}'.")

if __name__ == "__main__":
    input_csv_1 = "./Evaluation_Metrics/mmr_reranked_result.csv" 
    input_csv_2 = "./data/claims.csv"
    output_csv = "./Evaluation_Metrics/mmr_reranked_result.csv"
    merge_csvs_by_row(input_csv_1, input_csv_2, output_csv)
