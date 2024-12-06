import numpy as np
import pandas as pd
import pytrec_eval
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import T5Tokenizer, T5ForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

def prepare_pytrec_eval_input(ranked_list, scores, qrels, topic_id):
    topic_id = str(topic_id)
    ranked_list = [str(doc_id) for doc_id in ranked_list]
    results = {topic_id: {doc: float(score) for doc, score in zip(ranked_list, scores)}}
    topic_qrels = qrels.get(int(topic_id), {})
    qrels_dict = {topic_id: {str(doc): int(rel) for doc, rel in topic_qrels.items()}}
    
    return results, qrels_dict

def compute_metrics_pytrec_eval(results, qrels_dict):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict, 
        {"ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.50", "ndcg_cut.100", 
         "map_cut.5", "map_cut.10", "map_cut.20", "map_cut.50", "map_cut.100"}
    )
    metrics = evaluator.evaluate(results)
    return metrics

def bm25_retrieve(claim, corpus, top_n=100):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(claim.split())
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(i, scores[i]) for i in top_indices]

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
        scores = outputs.logits[:, 1].numpy()
    return list(np.argsort(scores)[::-1]), scores

def classify_with_t5_iteratively(claim, docs, tokenizer, model, target_support=75, target_contradict=75):
    support_docs = []
    contradict_docs = []
    neutral_docs = []
    support_indices = []
    contradict_indices = []
    neutral_indices = []
    placeholder_docs = []
    placeholder_indices = []
    placeholder_probs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    iter = 0
    for idx, doc in enumerate(docs):
        input_text = f"Classify claim: claim: {claim} document: {doc}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.softmax(dim=1)
            max_prob = torch.max(scores, dim=1)[0].item()
        if iter < 30:
            placeholder_probs.append(max_prob)
            placeholder_docs.append(doc)
            placeholder_indices.append(idx)
        elif iter == 30:
            mean_prob = torch.mean(torch.tensor(placeholder_probs)).item()
            lower_range = mean_prob * 0.99
            upper_range = mean_prob * 1.01
            for i in range(len(placeholder_docs)):
                if placeholder_probs[i] < lower_range:
                    contradict_docs.append(doc)
                    contradict_indices.append(idx)
                elif placeholder_probs[i] <= upper_range and placeholder_probs[i] >= lower_range:
                    neutral_docs.append(doc)
                    neutral_indices.append(idx)
                else:
                    support_docs.append(doc)
                    support_indices.append(idx)
            if max_prob < lower_range:
                contradict_docs.append(doc)
                contradict_indices.append(idx)
            elif max_prob >= lower_range and max_prob <= upper_range:
                neutral_docs.append(doc)
                neutral_indices.append(idx)
            else:
                support_docs.append(doc)
                support_indices.append(idx)
        else:
            if max_prob < lower_range:
                contradict_docs.append(doc)
                contradict_indices.append(idx)
            elif max_prob >= lower_range and max_prob <= upper_range:
                neutral_docs.append(doc)
                neutral_indices.append(idx)
            else:
                support_docs.append(doc)
                support_indices.append(idx)
        iter += 1
        if len(support_docs) == target_support and len(contradict_docs) == target_contradict:
            break

    remaining_support_needed = target_support - len(support_docs)
    remaining_contradict_needed = target_contradict - len(contradict_docs)

    if remaining_support_needed > 0:
        print("support", remaining_support_needed)
        support_docs.extend(neutral_docs[:remaining_support_needed])
        support_indices.extend(neutral_indices[:remaining_support_needed])

    if remaining_contradict_needed > 0:
        print("contradict", remaining_contradict_needed)
        contradict_docs.extend(neutral_docs[remaining_support_needed:remaining_support_needed + remaining_contradict_needed])
        contradict_indices.extend(neutral_indices[remaining_support_needed:remaining_support_needed + remaining_contradict_needed])
    return support_docs, contradict_docs, support_indices, contradict_indices


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
    
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_model = T5ForSequenceClassification.from_pretrained('t5-base', num_labels=3)

    tokenizer = AutoTokenizer.from_pretrained("NeuML/bert-small-cord19")
    model = AutoModelForSequenceClassification.from_pretrained("NeuML/bert-small-cord19")
    
    metrics = []
    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
        claim = row['claim']
        topic_id = int(row['topic_ip'])
        if topic_id not in qrels:
            continue
        
        bm25_results = bm25_retrieve(claim, cord19_docs, top_n=400)
        bm25_doc_ids = [cord19_docs.index[i] for i, _ in bm25_results]

        support_docs, contradict_docs, support_indices, contradict_indices = classify_with_t5_iteratively(
            claim,
            cord19_df.loc[bm25_doc_ids, 'title'].fillna('') + " " + cord19_df.loc[bm25_doc_ids, 'abstract'].fillna(''),
            t5_tokenizer,
            t5_model
        )
        support_doc_ids = [bm25_doc_ids[i] for i in support_indices]
        contradict_doc_ids = [bm25_doc_ids[i] for i in contradict_indices]
        support_rerank_indices, support_rerank_scores = rerank_with_biobert(
            claim, cord19_df.loc[support_doc_ids, 'title'].fillna('') + " " + cord19_df.loc[support_doc_ids, 'abstract'].fillna(''), tokenizer, model
        )
        support_reranked_doc_ids = [support_doc_ids[i] for i in support_rerank_indices]
        contradict_rerank_indices, contradict_rerank_scores = rerank_with_biobert(
            claim, cord19_df.loc[contradict_doc_ids, 'title'].fillna('') + " " + cord19_df.loc[contradict_doc_ids, 'abstract'].fillna(''), tokenizer, model
        )
        contradict_reranked_doc_ids = [contradict_doc_ids[i] for i in contradict_rerank_indices]
        metrics.append({
            "claim": claim,
            "support_reranked_docs": support_reranked_doc_ids,
            "contradict_reranked_docs": contradict_reranked_doc_ids,
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_csv = "./data/claims.csv"
    cord19_csv = "./data/processed_metadata.csv"
    qrels_csv = "./data/processed_qrels.csv"
    output_file = "./proposed_model/twoLists.csv"
    main(input_csv, cord19_csv, qrels_csv, output_file)
