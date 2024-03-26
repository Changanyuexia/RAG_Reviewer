# RARe

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

Note: we provide the example data, not the whole dataset.

### Data Processing

Only clean CRer. dataset: replaces newline ("\n") and tab ("\t") characters with a single space and consolidates multiple spaces into one.

## Retriever

We provide three retriver files: gpm_retrieval, normal_retrieval and dpr_retrieval. 

Our dpr code comes from: https://github.com/rizwan09/REDCODER/tree/main/SCODE-R
In gpm_retrieval, gpm_code2embedding.py is to generate the retrieval results of gpm retrieval. (CodeBERT + cosine similarity)
In normal_retrieval, code2embedding.py is to generate the retrieval results of normal retrieval. (CodeBERT + cosine similarity & gpm similarity)


## Generator

We provide two generative methods: fine-tuning and prompt tuning.

Fine-tuning code comes from: https://github.com/hiyouga/LLaMA-Factory/tree/main

And we provide the json file for prompt in fine-tuning.

The input of Generator should be the best performer of retriever. In this case, it is gpm_retriever.

prompt_review_generate.py is for generating prompts.

```bash

Top_k=1
Dataset="tf"
python prompt_review_generate.py \
    --Top_k $Top_k \
    --test_top10_path ../../retriever/gpm_retrieval/${Dataset}/test_top10.csv \
    --input_file input/${Dataset}/model_input_top_${Top_k}.jsonl
```

Then generate.py is for generating the target review.
```bash

Topk=1
Dataset="tf"
# Run the python script with arguments
python generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --input_file input/${Dataset}/model_input_top_${Topk}.jsonl \
    --output_file output/${Dataset}/generate_${Topk}_m7b.jsonl \
    --evaluation_file output/${Dataset}/eval_${Topk}_m7b.jsonl

```

