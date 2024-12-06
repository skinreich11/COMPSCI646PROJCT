import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def calculate_mmr(claim_embedding, doc_embeddings, relevance_scores, lambda_param=0.5, top_n=100):
    claim_embedding = normalize(claim_embedding.reshape(1, -1), axis=1)
    normalized_doc_embeddings = []
    for doc in doc_embeddings:
        try:
            if doc.shape[0] == 1:
                doc = doc.squeeze(0) 
            normalized_doc_embeddings.append(normalize(doc.reshape(1, -1), axis=1))
        except ValueError as e:
            print(f"Error processing document embedding: {e}. Skipping this document.")
            continue
    doc_embeddings = np.vstack(normalized_doc_embeddings)

    selected = []
    unselected = list(range(len(doc_embeddings)))

    while len(selected) < top_n and unselected:
        mmr_scores = []
        for doc_id in unselected:
            relevance = relevance_scores[doc_id]
            diversity = 0
            try:
                if selected:
                    diversity = max(
                        cosine_similarity(doc_embeddings[doc_id].reshape(1, -1), doc_embeddings[s].reshape(1, -1))[0][0]
                        for s in selected
                    )
            except ValueError:
                print(f"Dimension mismatch for document {doc_id}. Skipping.")
                continue

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((doc_id, mmr_score))

        if mmr_scores:
            best_doc = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_doc)
            unselected.remove(best_doc)
        else:
            break

    return selected


def get_document_embedding(doc, tokenizer, model):
    inputs_doc = tokenizer(doc, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs_doc = model(**inputs_doc)
    doc_embedding = outputs_doc.last_hidden_state.mean(dim=1).cpu().numpy()
    return doc_embedding

def main(input_csv, cord19_csv, output_file, lambda_param=0.5, top_n=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    combined_docs_df = pd.read_csv(input_csv)
    combined_docs_index = combined_docs_df.set_index('claim')
    cord19_df = pd.read_csv(cord19_csv)
    cord19_docs = (
        cord19_df['title'].fillna('').astype(str) + " " + cord19_df['abstract'].fillna('').astype(str)
    )
    cord19_docs.index = cord19_df['cord_uid']
    cord19_df.set_index('cord_uid', inplace=True)
    tokenizer = AutoTokenizer.from_pretrained("NeuML/bert-small-cord19")
    model = AutoModel.from_pretrained("NeuML/bert-small-cord19")
    results = []
    for _, row in tqdm(combined_docs_df.iterrows(), total=len(combined_docs_df), desc="Processing claims"):
        claim = row['claim']
        if claim not in combined_docs_index.index:
            print(f"Claim '{claim}' not found in input CSV. Skipping...")
            continue
        try:
            combined_doc_ids = eval(row['sorted_cord_uids'])
            combined_doc_scores = eval(row['sorted_scores'])
        except Exception as e:
            print(f"Error parsing sorted_cord_uids or sorted_scores for claim '{claim}': {e}")
            continue
        inputs_claim = tokenizer(claim, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            outputs_claim = model(**inputs_claim)
        claim_embedding = outputs_claim.last_hidden_state.mean(dim=1).numpy()
        doc_embeddings = []
        for doc_id in combined_doc_ids:
            doc_content = cord19_df.loc[doc_id, 'title'] + " " + cord19_df.loc[doc_id, 'abstract']
            doc_embedding = get_document_embedding(doc_content, tokenizer, model)
            doc_embeddings.append(doc_embedding)
        doc_embeddings = np.vstack(doc_embeddings)
        mmr_selected_indices = calculate_mmr(
            claim_embedding, doc_embeddings, combined_doc_scores, lambda_param=lambda_param, top_n=top_n
        )
        mmr_sorted_doc_ids = [combined_doc_ids[i] for i in mmr_selected_indices]
        results.append({
            'claim': claim,
            'sorted_cord_uids': mmr_sorted_doc_ids
        })
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_csv = "./proposed_model/singRankedListWithClass.csv"
    cord19_csv = "./data/processed_metadata.csv"
    output_file = "./Evaluation_Metrics/mmr_reranked_result.csv"
    main(input_csv, cord19_csv, output_file)