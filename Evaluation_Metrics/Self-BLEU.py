import os
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import pytrec_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.util import ngrams


# Function to perform BM25 retrieval
def bm25_retrieve(claim, corpus, top_n=100):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(claim.split())
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(i, scores[i]) for i in top_indices]


# Function to rerank using COVID-BERT
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
        scores = outputs.logits[:, 1].numpy()  # Assume binary classification, class 1 is relevance
    return list(np.argsort(scores)[::-1]), scores


# Function to calculate MMR scores
def calculate_mmr(claim_embedding, doc_embeddings, relevance_scores, lambda_param=0.5, top_n=100):
    claim_embedding = normalize(claim_embedding.squeeze(0).numpy().reshape(1, -1), axis=1)
    doc_embeddings = [
        normalize(doc.squeeze(0).numpy().reshape(1, -1), axis=1)
        if doc.shape[-1] == claim_embedding.shape[-1] else None
        for doc in doc_embeddings
    ]
    doc_embeddings = [doc for doc in doc_embeddings if doc is not None]

    selected = []
    unselected = list(range(len(doc_embeddings)))

    while len(selected) < top_n and unselected:
        mmr_scores = []
        for doc_id in unselected:
            relevance = relevance_scores[doc_id]
            diversity = max(
                cosine_similarity(doc_embeddings[doc_id], doc_embeddings[s])[0][0]
                for s in selected
            ) if selected else 0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((doc_id, mmr_score))

        if mmr_scores:
            best_doc = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_doc)
            unselected.remove(best_doc)
        else:
            break

    return selected


# Function to calculate Self-BLEU for diversity
def calculate_self_bleu(documents, n_grams=4, k_values=[5, 10, 20, 50, 100]):
    results = {}
    for k in k_values:
        if k > len(documents):
            continue
        selected_docs = documents[:k]

        # Prepare hypotheses and references for BLEU calculation
        hypotheses = []
        list_of_references = []

        for i, target_doc in enumerate(selected_docs):
            references = selected_docs[:i] + selected_docs[i + 1:]
            hypotheses.append(target_doc.split())
            list_of_references.append([ref.split() for ref in references])

        # Calculate corpus-level BLEU score
        smoothing = SmoothingFunction().method1
        score = corpus_bleu(list_of_references, hypotheses, weights=(1.0/n_grams,) * n_grams, smoothing_function=smoothing)
        results[f"Self_BLEU@{k}"] = score

    return results


# Function to prepare data for pytrec_eval
def prepare_pytrec_eval_input(ranked_list, scores, qrels, topic_id):
    topic_id = str(topic_id)
    results = {topic_id: {doc: float(score) for doc, score in zip(ranked_list, scores)}}
    topic_qrels = qrels.get(int(topic_id), {})
    qrels_dict = {topic_id: {doc: rel for doc, rel in topic_qrels.items()}}
    return results, qrels_dict


# Function to compute metrics using pytrec_eval
def compute_metrics_pytrec_eval(results, qrels_dict):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict,
        {"map_cut.5", "map_cut.10", "map_cut.20", "map_cut.50", "map_cut.100",
        "ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.50", "ndcg_cut.100",
         }
    )
    metrics = evaluator.evaluate(results)
    return metrics


# Main function to execute the pipeline
#def main(input_csv, cord19_csv, qrels_csv, output_file): # for all 53 claims
def main(input_csv, cord19_csv, qrels_csv, output_file, max_claims=5): # for 5 claims
    claims_df = pd.read_csv(input_csv)
    cord19_df = pd.read_csv(cord19_csv)
    cord19_df.set_index('cord_uid', inplace=True)
    cord19_docs = (
        cord19_df['title'].fillna('') + " " + cord19_df['abstract'].fillna('')
    ).tolist()

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

    results = []
    #for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"): # for all 53 claims
    for _, row in tqdm(claims_df.head(max_claims).iterrows(), total=max_claims, desc="Processing claims"): # Processing only the first `max_claims=5` rows
        claim = row['claim']
        topic_id = row['topic_ip']

        bm25_results = bm25_retrieve(claim, cord19_docs, top_n=100)
        bm25_doc_ids = [cord19_df.index[i] for i, _ in bm25_results]
        bm25_docs = [cord19_docs[i] for i, _ in bm25_results]

        rerank_indices, rerank_scores = rerank_with_biobert(
            claim, bm25_docs, tokenizer, model
        )
        reranked_doc_ids = [bm25_doc_ids[i] for i in rerank_indices]

        results_dict, qrels_dict = prepare_pytrec_eval_input(
            reranked_doc_ids, rerank_scores, qrels, topic_id
        )
        pytrec_metrics = compute_metrics_pytrec_eval(results_dict, qrels_dict)

        claim_embedding = tokenizer(claim, return_tensors="pt", truncation=True, padding="max_length", max_length=512)["input_ids"]
        doc_embeddings = [
            tokenizer(doc, return_tensors="pt", truncation=True, padding="max_length", max_length=512)["input_ids"]
            for doc in bm25_docs
        ]
        mmr_indices = calculate_mmr(claim_embedding, doc_embeddings, rerank_scores, top_n=100)
        mmr_docs = [bm25_docs[i] for i in mmr_indices]

        self_bleu_scores = calculate_self_bleu(mmr_docs)

        result_row = {
            'claim': claim,
            'topic_id': topic_id,
            **pytrec_metrics[str(topic_id)],
            **self_bleu_scores
        }
        results.append(result_row)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")



if __name__ == "__main__":
    input_csv = "S:/Code/COMPSCI646PROJCT/data/claims.csv"
    cord19_csv = "S:/Code/COMPSCI646PROJCT/data/processed_metadata.csv"
    qrels_csv = "S:/Code/COMPSCI646PROJCT/data/processed_qrels.csv"
    output_file = "S:/Code/COMPSCI646PROJCT/Evaluation_Metrics/final_results_mmr_selfBLEU.csv"
    main(input_csv, cord19_csv, qrels_csv, output_file)
