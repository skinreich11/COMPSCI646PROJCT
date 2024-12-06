import pandas as pd
from tqdm import tqdm
from typing import List
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def calculate_self_bleu(documents, n_grams=4, k_values=[5, 10, 20, 50, 100]):
    results = {}
    for k in k_values:
        if k > len(documents):
            continue
        selected_docs = documents[:k]
        hypotheses = []
        list_of_references = []
        for i, target_doc in enumerate(selected_docs):
            references = selected_docs[:i] + selected_docs[i + 1:]
            hypotheses.append(target_doc.split())
            list_of_references.append([ref.split() for ref in references])
        smoothing = SmoothingFunction().method1
        score = corpus_bleu(list_of_references, hypotheses, weights=(1.0/n_grams,) * n_grams, smoothing_function=smoothing)
        results[f"Self_BLEU@{k}"] = score
    return results

def process_documents(sorted_cord_uids: List[str], cord19_df: pd.DataFrame) -> List[str]:
    documents = []
    for doc_id in sorted_cord_uids:
        if doc_id in cord19_df.index:
            title = cord19_df.loc[doc_id, 'title']
            abstract = cord19_df.loc[doc_id, 'abstract']
            doc_text = f"{title} {abstract}".strip()
            if doc_text:
                documents.append(doc_text)
    return documents

def main():
    mmr_results_file = "./proposed_model/singRankedListWithClass.csv"
    cord19_csv = "./data/processed_metadata.csv"
    output_file = "./Evaluation_Metrics/final_nommr_ranklist_results_selfBLEU.csv"
    k_values = [5, 10, 20, 50, 100]
    mmr_results = pd.read_csv(mmr_results_file)
    cord19_df = pd.read_csv(cord19_csv)
    cord19_df.set_index('cord_uid', inplace=True)
    results = []
    for _, row in tqdm(mmr_results.iterrows(), total=len(mmr_results), desc="Processing claims"):
        claim = row['claim']
        topic_id = row['topic_ip']
        try:
            doc_ids = eval(row['sorted_cord_uids'])
            doc_scores = eval(row['sorted_scores'])
        except Exception as e:
            print(f"Error parsing sorted_cord_uids or sorted_scores for claim '{claim}': {e}")
            continue
        documents = process_documents(doc_ids, cord19_df)
        self_bleu_scores = calculate_self_bleu(documents, k_values=k_values)
        result = {
            "claim": claim,
            "topic_id": topic_id,
            "sorted_cord_uids": ",".join(doc_ids),
            "sorted_scores": ",".join(map(str, doc_scores))
        }
        for k in k_values:
            result.update({f'Self_BLEU@{k}': self_bleu_scores.get(f'Self_BLEU@{k}', 0.0)})
        results.append(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print("\nSummary Statistics:")
    for k in k_values:
        print(f"Average Self_BLEU@{k}: {results_df[f'Self_BLEU@{k}'].mean():.4f}")

if __name__ == "__main__":
    main()
