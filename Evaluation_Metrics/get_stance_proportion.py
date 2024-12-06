import pandas as pd
from collections import Counter

def calculate_cumulative_classification_proportions(csv_file, output_file, indices=[5, 10, 20, 50, 100]):
    df = pd.read_csv(csv_file)
    df['classification'] = df['classification'].apply(eval)
    allProortions = {ind: [] for ind in indices}
    for _, row in df.iterrows():
        totalNeu = 0.0
        totalSup = 0.0
        totalContra = 0.0
        average_proportions = {}
        classifications = row['classification']
        for ind, cl in enumerate(classifications):
            if cl == "neutral":
                totalNeu += 1
            elif cl == "support":
                totalSup += 1
            else:
                totalContra += 1
            if ind + 1 in indices:
                print(totalContra, totalNeu, totalSup)
                average_proportions[ind + 1] = {
                    "neutral": float(totalNeu) / float(ind + 1),
                    "contradict": float(totalContra) / float(ind + 1),
                    "support": float(totalSup) / float(ind + 1)}
        for ind in indices:
            allProortions[ind].append(average_proportions[ind])
    output_data = []
    for index in indices:
        output_data.append({
            "Index": index,
            "Neutral": sum(i["neutral"] for i in allProortions[index]) / len(allProortions[index]),
            "Contradict": sum(i["contradict"] for i in allProortions[index]) / len(allProortions[index]),
            "Support": sum(i["support"] for i in allProortions[index]) / len(allProortions[index]),
        })
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    csv_file_path = "./Evaluation_Metrics/mmr_reranked_result.csv"
    output_csv_file = "./Evaluation_Metrics/proposed_model_with_mmr_avarage_proportions.csv"
    calculate_cumulative_classification_proportions(csv_file_path, output_csv_file)
