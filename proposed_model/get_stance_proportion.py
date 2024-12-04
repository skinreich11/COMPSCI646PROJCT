import pandas as pd
from collections import Counter

def calculate_cumulative_classification_proportions(csv_file, output_file, indices=[5, 10, 20, 50, 100]):
    """
    Calculate and average the cumulative proportions of classifications up to specified indices.
    Save the results to a CSV file.

    Parameters:
        csv_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        indices (list): List of indices to evaluate (1-based).

    Returns:
        None
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the classification column is evaluated as lists
    df['classification'] = df['classification'].apply(eval)

    # Initialize dictionaries for cumulative proportions
    # Process each row
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

    # Prepare the results for saving
    output_data = []
    for index in indices:
        output_data.append({
            "Index": index,
            "Neutral": sum(i["neutral"] for i in allProortions[index]) / len(allProortions[index]),
            "Contradict": sum(i["contradict"] for i in allProortions[index]) / len(allProortions[index]),
            "Support": sum(i["support"] for i in allProortions[index]) / len(allProortions[index]),
        })

    # Save to a CSV file
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    csv_file_path = "./proposed_model/singRankedListWithClass2.csv"  # Replace with the path to your input CSV file
    output_csv_file = "./proposed_model/proposed_model_avarage_proportions.csv"  # Replace with your desired output file name
    calculate_cumulative_classification_proportions(csv_file_path, output_csv_file)
