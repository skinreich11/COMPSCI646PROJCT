import pandas as pd
import pytrec_eval
from tqdm import tqdm 


def prepare_pytrec_eval_input(ranked_list, scores, qrels, topic_id):
    topic_id = str(topic_id)
    ranked_list = [str(doc_id) for doc_id in eval(ranked_list)]
    
    # Create the results dictionary for pytrec_eval
    results = {topic_id: {doc: float(score) for doc, score in zip(ranked_list, eval(scores))}}
    
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

def main():
    qrels_df = pd.read_csv("./data/processed_qrels.csv")
    qrels = {}
    for _, row in qrels_df.iterrows():
        topic = int(row['topic_ip'])
        cord_uid = row['cord_uid']
        relevance = int(row['relevance'])
        if topic not in qrels:
            qrels[topic] = {}
        qrels[topic][cord_uid] = relevance
    metrics = []
    model = pd.read_csv("./proposed_model/singRankedListWithClass.csv")
    for _, row in tqdm(model.iterrows(), total=len(model)):
        claim = row['claim']
        topic_id = int(row['topic_ip'])
        
        # Skip topics without qrels
        if topic_id not in qrels:
            continue
    
        results, qrels_dict = prepare_pytrec_eval_input(
                row["sorted_cord_uids"],
                row["sorted_scores"],
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
    average_metrics = metrics_df.mean(numeric_only=True)  # Calculate mean for numeric columns
    average_metrics.to_csv("./proposed_model/average_results.csv", header=["average"], index_label="metric")
    print(f"Average results saved to ./proposed_model/average_results.csv")
main()
