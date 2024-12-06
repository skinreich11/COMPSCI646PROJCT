import os
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import pytrec_eval

def BM25(claim, corpus, top_n=100):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(claim.split())
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(i, scores[i]) for i in top_indices]

def COVIDBERT(claim, docs, tokenizer, model):
    inputs = tokenizer(
        [f"{claim} [SEP] {doc}" for doc in docs], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[:, 1].numpy()
    return list(np.argsort(scores)[::-1]), scores

def formatForPytrec(ranked_list, scores, qrels, topic_id):
    topic_id = str(topic_id)
    ranked_list = [str(doc_id) for doc_id in ranked_list]
    results = {topic_id: {doc: float(score) for doc, score in zip(ranked_list, scores)}}
    topic_qrels = qrels.get(int(topic_id), {})
    qrels_dict = {topic_id: {str(doc): int(rel) for doc, rel in topic_qrels.items()}}
    return results, qrels_dict

def computeScores(results, qrels_dict):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict, 
        {"ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.50", "ndcg_cut.100", 
         "map_cut.5", "map_cut.10", "map_cut.20", "map_cut.50", "map_cut.100"}
    )
    metrics = evaluator.evaluate(results)
    return metrics

def main(input_csv, cord19_csv, qrels_csv, output_file):
    claims_df = pd.read_csv(input_csv)
    cord19_df = pd.read_csv(cord19_csv)
    cord19_docs = (
        cord19_df['title'].fillna('').astype(str) + " " + cord19_df['abstract'].fillna('').astype(str)
    )
    cord19_docs.index = cord19_df['cord_uid']
    cord19_df.set_index('cord_uid', inplace=True)
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
    metrics = []
    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
        claim = row['claim']
        topic_id = int(row['topic_ip'])
        if topic_id not in qrels:
            continue
        bm25_results = BM25(claim, cord19_docs, top_n=100)
        bm25_doc_ids = [cord19_docs.index[i] for i, _ in bm25_results]
        rerank_indices, rerank_scores = COVIDBERT(
            claim,
            cord19_df.loc[bm25_doc_ids, 'title'].fillna('') + " " + cord19_df.loc[bm25_doc_ids, 'abstract'].fillna(''),
            tokenizer,
            model
        )
        reranked_doc_ids = [bm25_doc_ids[i] for i in rerank_indices]
        results, qrels_dict = formatForPytrec(
            reranked_doc_ids,
            rerank_scores,
            qrels,
            topic_id
        )
        claim_metrics = computeScores(results, qrels_dict)
        claim_metrics[str(topic_id)]['claim'] = claim
        claim_metrics[str(topic_id)]['topic_id'] = topic_id
        metrics.append(claim_metrics[str(topic_id)])
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_csv = "./data/claims.csv"
    cord19_csv = "./data/processed_metadata.csv"
    qrels_csv = "./data/processed_qrels.csv"
    output_file = "./baseline/results.csv"
    main(input_csv, cord19_csv, qrels_csv, output_file)
