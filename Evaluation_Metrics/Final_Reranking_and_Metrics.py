import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import pytrec_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.util import ngrams

# Function to calculate MMR scores
def calculate_mmr(claim_embedding, doc_embeddings, relevance_scores, lambda_param=0.5, top_n=100):
    """
    Calculate MMR scores and return top-ranked documents.
    :param claim_embedding: The embedding of the claim (query)
    :param doc_embeddings: The embeddings of the documents
    :param relevance_scores: The relevance scores from BioBERT
    :param lambda_param: The trade-off between relevance and diversity (0 <= lambda_param <= 1)
    :param top_n: Number of documents to return
    :return: Ranked list of document indices
    """
    # Normalize embeddings
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
            diversity = 0
            try:
                if selected:
                    diversity = max(
                        cosine_similarity(doc_embeddings[doc_id], doc_embeddings[s])[0][0]
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

#@TODO: Copied from Diversity metrics calculator, import and reference later
def process_documents(doc_ids: List[str], cord19_df: pd.DataFrame) -> List[str]:
    """Process documents from CORD-19 dataset"""
    documents = []
    print(f"\nProcessing {len(doc_ids)} documents...")
    
    for doc_id in doc_ids:
        try:
            if doc_id in cord19_df.index:
                title = str(cord19_df.loc[doc_id, 'title']).strip()
                abstract = str(cord19_df.loc[doc_id, 'abstract']).strip()
                
                doc_text = f"{title} {abstract}".strip()
                if doc_text:
                    documents.append(doc_text)
                    if len(documents) <= 3:
                        print(f"Sample document {len(documents)}: {doc_text[:100]}...")
            else:
                print(f"Document ID {doc_id} not found in CORD-19 dataset")
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            continue
    
    print(f"Successfully processed {len(documents)} valid documents")
    return documents

# Function to calculate MMR scores and include in metrics
def calculate_mmr_score_and_update_metrics(claim, claim_embedding, doc_embeddings, reranked_scores, reranked_doc_ids, top_n=10, lambda_param=0.5):
    # Select documents using MMR
    mmr_selected_indices = calculate_mmr(claim_embedding, doc_embeddings, reranked_scores, lambda_param=lambda_param, top_n=top_n)

    # Calculate average MMR relevance score
    mmr_selected_scores = [reranked_scores[i] for i in mmr_selected_indices]
    avg_mmr_score = np.mean(mmr_selected_scores) if mmr_selected_scores else 0.0

    # Get MMR-selected document IDs
    mmr_selected_doc_ids = [reranked_doc_ids[i] for i in mmr_selected_indices]

    return avg_mmr_score, mmr_selected_doc_ids

# Function to calculate Self-BLEU for diversity
#@TODO: Copied from Diversity metrics calculator, import and reference later
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

# Main function to calculate all metrics for final reranking list
def main(input_csv, cord19_csv, qrels_csv, output_file, max_claims=5):
    # Load input CSV reranked file and convert other files to datasets
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

    # @TODO: Change this to run from 5 claims to all claims by commenting first and uncommenting 2nd
    for _, row in tqdm(input_csv.head(max_claims).iterrows(), total=max_claims, desc="Processing claims"):
    # for _, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
        claim = row['claim']
        topic_id = int(row['topic_ip'])
        
        # Skip topics without qrels
        if topic_id not in qrels:
            continue

        # @TODO: Stav @Wentao : Update final_reranked_doc_ids and final_rerank_scores
        final_reranked_doc_ids = []
        final_rerank_scores = []

        # Prepare data for pytrec_eval
        results, qrels_dict = prepare_pytrec_eval_input(
            final_reranked_doc_ids,
            final_rerank_scores,
            qrels,
            topic_id
        )
        
        # MMR calculation
        claim_embedding = tokenizer(claim, return_tensors="pt", truncation=True, padding="max_length", max_length=512)["input_ids"]
        doc_embeddings = [
            tokenizer(doc, return_tensors="pt", truncation=True, padding="max_length", max_length=512)["input_ids"]
            for doc in cord19_df.loc[final_reranked_doc_ids, 'title'].fillna('') + " " + cord19_df.loc[final_reranked_doc_ids, 'abstract'].fillna('')
        ]
        avg_mmr_score, mmr_selected_doc_ids = calculate_mmr_score_and_update_metrics(
            claim, claim_embedding, doc_embeddings, final_rerank_scores, final_reranked_doc_ids, top_n=100
        )

        # Calculate self-BLEU for MMR selected documents
        documents = process_documents(mmr_selected_doc_ids, cord19_df)
        self_bleu_scores = calculate_self_bleu(documents)
        
        # Compute evaluation metrics using pytrec_eval
        claim_metrics = compute_metrics_pytrec_eval(results, qrels_dict)

        # save results to csv
        result_row = {
            'claim': claim,
            'topic_id': topic_id,
            **claim_metrics[str(topic_id)],
            'avg_mmr_score': avg_mmr_score,
            **self_bleu_scores
        }
        results.append(result_row)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # @TODO: Modify below file names to pass correct file inputs
    RRF_Results_csv = "./Evaluation_Metrics/fusion_results.csv" # Input final fusion results csv file
    cord19_csv = "./data/processed_metadata.csv"  # CORD-19 dataset CSV file
    qrels_csv = "./data/processed_qrels.csv"  # Qrels file in CSV format
    output_file = "./Evaluation_Metrics/final_results_mmr_selfBLEU.csv"  # Output file to save evaluation metrics
    
    main(RRF_Results_csv, cord19_csv, qrels_csv, output_file)
