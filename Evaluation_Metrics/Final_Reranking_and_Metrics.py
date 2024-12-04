import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
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
    claim_embedding = normalize(claim_embedding.reshape(1, -1), axis=1)
    
    # Normalize document embeddings and check shape compatibility
    normalized_doc_embeddings = []
    for doc in doc_embeddings:
        try:
            # Ensure that doc is not None and has the right shape
            if doc.shape[0] == 1:  # Checking if batch size is 1
                doc = doc.squeeze(0)  # Remove batch dimension of size 1
            # Reshape and normalize the document embedding
            normalized_doc_embeddings.append(normalize(doc.reshape(1, -1), axis=1))
        except ValueError as e:
            # Handle dimension mismatch error
            print(f"Error processing document embedding: {e}. Skipping this document.")
            continue

    # Now we have normalized document embeddings
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

# Function to compute document embeddings
def get_document_embedding(doc, tokenizer, model):
    inputs_doc = tokenizer(doc, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs_doc = model(**inputs_doc)
    # Use the mean of the last hidden state as the document embedding
    doc_embedding = outputs_doc.last_hidden_state.mean(dim=1).cpu().numpy()
    return doc_embedding

# Main function to get final reranking list of documents
def main(input_csv, cord19_csv, output_file, max_claims=5, lambda_param=0.5, top_n=100):
    # Set device to GPU if available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load input CSV combined file and convert other files to datasets
    combined_docs_df = pd.read_csv(input_csv)
    combined_docs_index = combined_docs_df.set_index('claim')  # Index by claim for fast lookup

    # Load CORD-19 dataset
    cord19_df = pd.read_csv(cord19_csv)
    cord19_docs = (
        cord19_df['title'].fillna('').astype(str) + " " + cord19_df['abstract'].fillna('').astype(str)
    )
    cord19_docs.index = cord19_df['cord_uid']
    cord19_df.set_index('cord_uid', inplace=True)

    # Load model and tokenizer for claim/document embeddings
    tokenizer = AutoTokenizer.from_pretrained("NeuML/bert-small-cord19")
    model = AutoModel.from_pretrained("NeuML/bert-small-cord19")  # Change to AutoModel for hidden state

    # Initialize output data
    results = []

    # Process claims
    for _, row in tqdm(combined_docs_df.head(max_claims).iterrows(), total=max_claims, desc="Processing claims"):
        claim = row['claim']
        
        # Check if claim exists in input_csv
        if claim not in combined_docs_index.index:
            print(f"Claim '{claim}' not found in input CSV. Skipping...")
            continue

        # Pull corresponding final_reranked_doc_ids and scores
        try:
            combined_doc_ids = eval(row['sorted_cord_uids'])
            combined_doc_scores = eval(row['sorted_scores'])
        except Exception as e:
            print(f"Error parsing sorted_cord_uids or sorted_scores for claim '{claim}': {e}")
            continue

        # Encode claim and documents
        # Claim embedding
        inputs_claim = tokenizer(claim, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            outputs_claim = model(**inputs_claim)
        claim_embedding = outputs_claim.last_hidden_state.mean(dim=1).numpy()  # Average across token embeddings

        # Document embeddings
        doc_embeddings = []
        for doc_id in combined_doc_ids:
            doc_content = cord19_df.loc[doc_id, 'title'] + " " + cord19_df.loc[doc_id, 'abstract']
            doc_embedding = get_document_embedding(doc_content, tokenizer, model)
            doc_embeddings.append(doc_embedding)

        # Convert to numpy array
        doc_embeddings = np.vstack(doc_embeddings)

        # Calculate MMR and re-rank documents
        mmr_selected_indices = calculate_mmr(
            claim_embedding, doc_embeddings, combined_doc_scores, lambda_param=lambda_param, top_n=top_n
        )
        mmr_sorted_doc_ids = [combined_doc_ids[i] for i in mmr_selected_indices]

        # Append results to metrics
        results.append({
            'claim': claim,
            'sorted_cord_uids': mmr_sorted_doc_ids
        })

    # Write metrics to output file
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Modify below file names to pass correct file inputs
    input_csv = "S:/Code/COMPSCI646PROJCT/proposed_model/singRankedListWithClass.csv"
    cord19_csv = "S:/Code/COMPSCI646PROJCT/data/processed_metadata.csv"  # CORD-19 dataset CSV file
    output_file = "S:/Code/COMPSCI646PROJCT/Evaluation_Metrics/mmr_reranked_result.csv"  # Output file to save evaluation metrics
    
    main(input_csv, cord19_csv, output_file)