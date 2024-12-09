# COMPSCI646PROJCT

## Overview
This project aims to enhance standard two-stage IR systems with T5 and MMR for biomedical hypothesis research. The system retrieves and ranks documents to provide a balanced set of supporting and contradicting evidence for given claims.

## Directory Structure
- `baseline/`: Contains scripts and data for evaluating the baseline model.
  - `evaluate_baseline.py`: Script to evaluate the baseline model.
  - `baseline_diversity_results.csv`: Results of diversity metrics for the baseline model.
  - `results.csv`: Evaluation results of the baseline model.
- `data/`: Contains datasets and scripts for data processing.
  - `claims.csv`: Claims dataset.
  - `filtered_cord_uids_metadata.txt`: Metadata for filtered CORD-19 documents.
  - `process_metadata.py`: Script to process metadata.
  - `process_qrels.py`: Script to process qrels.
  - `getClaims.py`: Script to generate claims dataset.
- `Evaluation_Metrics/`: Contains scripts for evaluating the proposed model.
  - `add_scores.py`: Script to add classification scores to CORD-19 UIDs.
  - `Final_Reranking_and_Metrics.py`: Script to re-rank documents using MMR and compute evaluation metrics.
  - `get_relevance.py`: Script to compute relevance metrics.
- `proposed_model/`: Contains scripts and data for the proposed model.
  - `evaluate_model.py`: Script to evaluate the proposed model.
  - `singRankedListWithClass.csv`: Ranked list of documents with classifications.
  - `twoLists.csv`: Combined list of supporting and contradicting documents.
- `RRF/`: Contains scripts for Reciprocal Rank Fusion (RRF).
  - `RRF.py`: Script to process results using RRF.
- `avarage_results.csv`: Average results of evaluation metrics.
- `diversity_results.csv`: Diversity results for the proposed model.
- `main.tex`: LaTeX file for the project report.
- `Project_Milestone1.tex`: LaTeX file for the project milestone report.

## Setup Instructions
1. Download the HealthVer dataset from HealthVer GitHub and run `getClaims.py` to generate `claims.csv`.
2. Download the 2020-07-16 version of the CORD-19 dataset from CORD-19 GitHub.
3. Download the qrels file from NIST COVID Submit.

## Running the Baseline Model
1. Run `process_metadata.py` to process the metadata.
2. Run `process_qrels.py` to process the qrels.
3. Run `evaluate_baseline.py` to retrieve documents using the baseline model.
4. Run `Diversity Metrics Calculator.py` to get evaluation metrics results for the baseline model.

## Running the Proposed Model
1. Run `evaluate_model.py` to get lists of supporting and contradicting documents for each claim.
2. Run `combine_lists.py` to combine these lists into a single list for each claim.
3. Run `Final_Reranking_and_Metrics.py` to re-rank documents using MMR.
4. Run `Self-BLEU.py` with the input file `mmr_reranked_result.csv` to get self-BLEU scores for documents ranked using MMR.

## Evaluation Metrics
- `ndcg@k`: Normalized Discounted Cumulative Gain at k.
- `map@k`: Mean Average Precision at k.
- `stance_support@k`: Proportion of supporting documents at k.
- `stance_contradict@k`: Proportion of contradicting documents at k.
- `stance_neutral@k`: Proportion of neutral documents at k.
- `inverse_simpson@k`: Inverse Simpson Index for diversity at k.

## Contact
For any questions or issues, please contact the project maintainers:
- Stav Kinreich: [skinreich@umass.edu](mailto:skinreich@umass.edu)
- Sreevidya Bollineni: [sreevidyabol@umass.edu](mailto:sreevidyabol@umass.edu)
- Wentao Ma: [wentaoma@umass.edu](mailto:wentaoma@umass.edu)