import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load COVID-BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("NeuML/bert-small-cord19")
model = AutoModelForSequenceClassification.from_pretrained("NeuML/bert-small-cord19")

def calculate_score(claim, document):
    """
    Calculate the similarity score between the claim and the document using COVID-BERT.
    """
    inputs = tokenizer(
        f"claim: {claim} document: {document}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.softmax(dim=1)[0, 1].item()  # Assuming relevance is class 1
    return score

def normalize_scores(scores):
    """
    Normalize a list of scores between 0 and 1.
    """
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:  # Handle edge case where all scores are the same
        return [0.5] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

# Load input CSVs
claims_file = "./proposed_model/twoLists.csv"  # Replace with your file path
metadata_file = "./data/processed_metadata.csv"  # Replace with your file path
df_claims = pd.read_csv(claims_file)
df_metadata = pd.read_csv(metadata_file)

# Create a dictionary mapping cord_uid to concatenated title + abstract
df_metadata['document'] = (
    df_metadata['title'].fillna('') + " " + df_metadata['abstract'].fillna('')
)
cord_uid_to_doc = df_metadata.set_index('cord_uid')['document'].to_dict()

# Process each row in the claims file
result = []
for _, row in tqdm(df_claims.iterrows(), total=len(df_claims)):
    claim = row['claim']
    support_uids = eval(row['support_reranked_docs'])  # Ensure list type
    contradict_uids = eval(row['contradict_reranked_docs'])  # Ensure list type

    # Match cord_uids to documents
    support_docs = [cord_uid_to_doc[uid] for uid in support_uids if uid in cord_uid_to_doc]
    contradict_docs = [cord_uid_to_doc[uid] for uid in contradict_uids if uid in cord_uid_to_doc]

    # Score documents
    support_scores = [calculate_score(claim, doc) for doc in support_docs]
    contradict_scores = [calculate_score(claim, doc) for doc in contradict_docs]

    # Normalize scores
    support_scores_normalized = normalize_scores(support_scores)
    contradict_scores_normalized = normalize_scores(contradict_scores)

    # Combine lists
    combined_docs = support_uids + contradict_uids
    combined_scores = support_scores_normalized + contradict_scores_normalized

    # Sort documents by scores in descending order
    sorted_docs_scores = sorted(
        zip(combined_docs, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Add to results
    result.append({
        'claim': claim,
        'sorted_cord_uids': [doc for doc, score in sorted_docs_scores],
        'sorted_scores': [score for doc, score in sorted_docs_scores]
    })

# Save results to a new CSV
output_file = "./proposed_model/singRankedList.csv"  # Replace with your desired output file path
pd.DataFrame(result).to_csv(output_file, index=False)
