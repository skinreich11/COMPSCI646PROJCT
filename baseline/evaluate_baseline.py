import os
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import pytrec_eval

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

# Prepare data for pytrec_eval
def prepare_pytrec_eval_input(ranked_list, scores, qrels, topic_id):
    topic_id = str(topic_id)
    ranked_list = [str(doc_id) for doc_id in ranked_list]
    
    # Create the results dictionary for pytrec_eval
    results = {topic_id: {doc: float(score) for doc, score in zip(ranked_list, scores)}}
    
    # Create the qrels dictionary for pytrec_eval
    topic_qrels = qrels.get(int(topic_id), {})
    qrels_dict = {topic_id: {str(doc): int(rel) for doc, rel in topic_qrels.items()}}
    
    return results, qrels_dict

# Compute evaluation metrics using pytrec_eval
def compute_metrics_pytrec_eval(results, qrels_dict):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict, 
        {"ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.50", "ndcg_cut.100", 
         "map_cut.5", "map_cut.10", "map_cut.20", "map_cut.50", "map_cut.100"}
    )
    metrics = evaluator.evaluate(results)
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
    cord19_df.set_index('cord_uid', inplace=True)
    
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
    
    tokenizer = AutoTokenizer.from_pretrained("NeuML/bert-small-cord19")
    model = AutoModelForSequenceClassification.from_pretrained("NeuML/bert-small-cord19")

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
        bm25_doc_ids = [cord19_docs.index[i] for i, _ in bm25_results]

        # BioBERT reranking
        rerank_indices, rerank_scores = rerank_with_biobert(
            claim,
            cord19_df.loc[bm25_doc_ids, 'title'].fillna('') + " " + cord19_df.loc[bm25_doc_ids, 'abstract'].fillna(''),
            tokenizer,
            model
        )
        reranked_doc_ids = [bm25_doc_ids[i] for i in rerank_indices]

        # Prepare data for pytrec_eval
        results, qrels_dict = prepare_pytrec_eval_input(
            reranked_doc_ids,
            rerank_scores,
            qrels,
            topic_id
        )
        
        # Compute evaluation metrics using pytrec_eval
        claim_metrics = compute_metrics_pytrec_eval(results, qrels_dict)
        claim_metrics[str(topic_id)]['claim'] = claim
        claim_metrics[str(topic_id)]['topic_id'] = topic_id
        metrics.append(claim_metrics[str(topic_id)])
    
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
