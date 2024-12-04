import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForSequenceClassification

def classify_with_t5_add_classification(input_csv, output_csv, tokenizer, model):
    # Read the input CSV
    df = pd.read_csv(input_csv)
    metadata_file = "./data/processed_metadata.csv"  # Replace with your file path
    df_metadata = pd.read_csv(metadata_file)
    df_metadata['document'] = (
    df_metadata['title'].fillna('') + " " + df_metadata['abstract'].fillna('')
    )
    cord_uid_to_doc = df_metadata.set_index('cord_uid')['document'].to_dict()
    # Ensure the device is set for T5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize the classification column
    df['classification'] = [[] for _ in range(len(df))]
    
    # Process each row in the CSV
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        claim = row['claim']
        sorted_cord_uids = eval(row['sorted_cord_uids'])
        classifications = []
        placeholder_probs = []
        placeholder_docs = []
        lower_range, upper_range = None, None
        
        # Process each document
        for iter, cord_uid in enumerate(sorted_cord_uids):
            doc = cord_uid_to_doc[cord_uid]
            input_text = f"Classify claim: claim: {claim} document: {doc}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits.softmax(dim=1)
                max_prob = torch.max(scores, dim=1)[0].item()
                predicted_class = torch.argmax(scores, dim=1).item()
            
            # Collect placeholder probabilities for initial iterations
            if iter < 100:
                placeholder_probs.append(max_prob)
                placeholder_docs.append(predicted_class)
            elif iter == 100:
                # Calculate range
                mean_prob = torch.mean(torch.tensor(placeholder_probs)).item()
                lower_range = mean_prob * 0.995
                upper_range = mean_prob * 1.005
                
                # Classify the initial 30 documents based on range
                for prob, pred_class in zip(placeholder_probs, placeholder_docs):
                    if prob < lower_range:
                        classifications.append("contradict")
                    elif lower_range <= prob <= upper_range:
                        classifications.append("neutral")
                    else:
                        classifications.append("support")
                
                # Classify the current document
                if max_prob < lower_range:
                    classifications.append("contradict")
                elif lower_range <= max_prob <= upper_range:
                    classifications.append("neutral")
                else:
                    classifications.append("support")
            else:
                # Classify subsequent documents
                if max_prob < lower_range:
                    classifications.append("contradict")
                elif lower_range <= max_prob <= upper_range:
                    classifications.append("neutral")
                else:
                    classifications.append("support")
        
        # Add the classifications list to the dataframe
        df.at[idx, 'classification'] = classifications
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForSequenceClassification.from_pretrained("t5-base", num_labels=3)

classify_with_t5_add_classification("./proposed_model/singRankedList.csv", "./proposed_model/singRankedListWithClass2.csv", tokenizer, model)