import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("NeuML/bert-small-cord19")
model = AutoModelForSequenceClassification.from_pretrained("NeuML/bert-small-cord19")

def calculate_score(claim, document):
    inputs = tokenizer(
        f"claim: {claim} document: {document}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.softmax(dim=1)[0, 1].item()
    return score

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]
claims_file = "./proposed_model/twoLists.csv"
metadata_file = "./data/processed_metadata.csv"
df_claims = pd.read_csv(claims_file)
df_metadata = pd.read_csv(metadata_file)
df_metadata['document'] = (
    df_metadata['title'].fillna('') + " " + df_metadata['abstract'].fillna('')
)
cord_uid_to_doc = df_metadata.set_index('cord_uid')['document'].to_dict()
result = []
for _, row in tqdm(df_claims.iterrows(), total=len(df_claims)):
    claim = row['claim']
    support_uids = eval(row['support_reranked_docs'])
    contradict_uids = eval(row['contradict_reranked_docs'])
    support_docs = [cord_uid_to_doc[uid] for uid in support_uids if uid in cord_uid_to_doc]
    contradict_docs = [cord_uid_to_doc[uid] for uid in contradict_uids if uid in cord_uid_to_doc]
    support_scores = [calculate_score(claim, doc) for doc in support_docs]
    contradict_scores = [calculate_score(claim, doc) for doc in contradict_docs]
    support_scores_normalized = normalize_scores(support_scores)
    contradict_scores_normalized = normalize_scores(contradict_scores)
    combined_docs = support_uids + contradict_uids
    combined_scores = support_scores_normalized + contradict_scores_normalized
    sorted_docs_scores = sorted(
        zip(combined_docs, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )
    result.append({
        'claim': claim,
        'sorted_cord_uids': [doc for doc, score in sorted_docs_scores],
        'sorted_scores': [score for doc, score in sorted_docs_scores]
    })

output_file = "./proposed_model/singRankedList.csv"
pd.DataFrame(result).to_csv(output_file, index=False)
