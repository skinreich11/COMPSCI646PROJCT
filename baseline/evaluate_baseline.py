import os
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import ndcg_score
import torch
import numpy as np
from tqdm import tqdm

# Helper function for BM25 retrieval
def bm25_retrieve(claim, corpus, top_n=100):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(claim.split())
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(i, scores[i]) for i in top_indices]

# Helper function for BioBERT reranking
def rerank_with_biobert(claim, docs, tokenizer, model):
    inputs = tokenizer(
        [f"{claim} [SEP] {doc}" for doc in docs], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[:, 1].numpy()  # Assume binary classification, 1 is relevance
    return list(np.argsort(scores)[::-1]), scores

# Compute evaluation metrics
def compute_metrics(ranked_list, qrels, k_values):
    metrics = {}
    relevance_scores = np.array([qrels.get(doc, 0) for doc in ranked_list])  # Default relevance is 0
    
    # Calculate NDCG@k
    for k in k_values:
        top_k = ranked_list[:k]
        top_k_relevance = [qrels.get(doc, 0) for doc in top_k]

        # Create an ideal ranking for the top-k documents based on available qrels
        ideal_relevance = sorted(qrels.values(), reverse=True)[:k]

        # Calculate NDCG@k
        metrics[f"NDCG@{k}"] = ndcg_score([ideal_relevance], [top_k_relevance])

    # Calculate MAP@k
    for k in k_values:
        top_k_relevance = relevance_scores[:k]
        cumulative_relevance = [
            rel / (i + 1) if rel > 0 else 0 for i, rel in enumerate(top_k_relevance)
        ]
        metrics[f"MAP@{k}"] = np.sum(cumulative_relevance) / min(k, len(qrels))
    
    return metrics

# Main pipeline
def main(input_csv, cord19_csv, qrels_csv, output_file):
    # Load input CSV
    claims_df = pd.read_csv(input_csv)
    
    # Load CORD-19 dataset
    cord19_df = pd.read_csv(cord19_csv)
    cord19_docs = (
    cord19_df['title'].fillna('').astype(str) + " " + cord19_df['abstract'].fillna('').astype(str)
    )
    cord19_docs.index = cord19_df['cord_uid']
    
    # Load relevance judgments (qrels)
    qrels_df = pd.read_csv(qrels_csv)
    qrels = {}
    for _, row in qrels_df.iterrows():
        topic = int(row['topic_ip'])
        cord_uid = row['cord_uid']
        relevance = int(row['relevance'])
        if topic not in qrels:
            qrels[topic] = {}
        qrels[topic][cord_uid] = relevance
    
    # Load BioBERT
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Metrics collection
    metrics = []

    # Process each claim
    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
        claim = row['claim']
        topic_id = int(row['topic_ip'])
        
        # Skip topics without qrels
        if topic_id not in qrels:
            continue
        
        # BM25 retrieval
        bm25_results = bm25_retrieve(claim, cord19_docs, top_n=100)
        bm25_doc_ids = [cord19_df.index[i] for i, _ in bm25_results]
        bm25_scores = [score for _, score in bm25_results]

        # BioBERT reranking
        rerank_indices, rerank_scores = rerank_with_biobert(
            claim, 
            [cord19_docs[doc_id] for doc_id in bm25_doc_ids],
            tokenizer, 
            model
        )
        reranked_doc_ids = [bm25_doc_ids[i] for i in rerank_indices]

        # Compute metrics
        claim_metrics = compute_metrics(
            reranked_doc_ids, 
            qrels[topic_id],  # Pass qrels for the specific topic
            k_values=[5, 10, 20, 50, 100]
        )
        claim_metrics['claim'] = claim
        claim_metrics['topic_id'] = topic_id
        metrics.append(claim_metrics)
    
    # Save results to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    input_csv = "./data/claims.csv"  # Input file with claim, topic_ip columns
    cord19_csv = "./data/processed_metadata.csv"  # CORD-19 dataset CSV file
    qrels_csv = "./data/processed_qrels.csv"  # Qrels file in CSV format
    output_file = "./baseline/results.csv"  # Output file to save evaluation metrics

    main(input_csv, cord19_csv, qrels_csv, output_file)
