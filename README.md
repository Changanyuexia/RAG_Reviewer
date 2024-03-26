# Project Name

This replication package is created for the paper "Code Review Automation using Retrieval Augmented Generation"

## Dataset
| Dataset | Train   | Test   | Val    | Total   |
|---------|---------|--------|--------|---------|
| Tuf.    | 134,239 | 16,780 | 16,780 | 167,799 |
| CRer.   | 117,740 | 10,170 | 10,320 | 138,230 |

### Complete datasets downloads from: 
Tuf. Dataset: https://github.com/RosaliaTufano/code_review_automation 

CRer. Dataset: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer

### Data Structure

Each datasets have jsonl and csv form. 

jsonl is for dpr retrieval.

csv is for gpm and normal retrieval.

### Data Processing

Only clean CRer. dataset: replaces newline ("\n") and tab ("\t") characters with a single space and consolidates multiple spaces into one.

## Retriever

We provide three retriver files: gpm_retrieval, normal_retrieval and dpr_retrieval. 

Our dpr code comes from: https://github.com/rizwan09/REDCODER/tree/main/SCODE-R


## Generator

We provide two generative methods: fine-tuning and prompt tuning.

Fine-tuning code comes from: https://github.com/hiyouga/LLaMA-Factory/tree/main

prompt_review_generate.py is for generating prompts.

```bash

Top_k=0
Dataset="tf"
python prompt_review_generate.py \
    --Top_k $Top_k \
    --test_top10_path dataset/${Dataset}/gpm_retrieval/test_top10.csv \
    --input_file input/${Dataset}/model_input_top_${Top_k}.jsonl \



