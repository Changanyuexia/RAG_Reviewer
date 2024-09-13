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


## Retriever

We provide three retriver files: gpm_retrieval, normal_retrieval and dpr_retrieval. 

Our dpr code comes from: https://github.com/rizwan09/REDCODER/tree/main/SCODE-R
In gpm_retrieval, gpm_code2embedding.py is to generate the retrieval results of gpm retrieval. (CodeBERT + cosine similarity)
In normal_retrieval, code2embedding.py is to generate the retrieval results of normal retrieval. (CodeBERT + cosine similarity & gpm similarity)


## Generator

We provide two generative methods: fine-tuning and direct inference.
We apply the framework from: https://github.com/hiyouga/LLaMA-Factory/tree/main
And we provide the json file for generator input.

The difference of templates of fine-tuning and direct inference is finetuning need training, and direct inference does not. Here is an example template for fine-tuing. steps.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/llama3_train_tf.yaml
llamafactory-cli train examples/train_lora/llama3_predict_tf.yaml

```

Here is an example template for direct inference.
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3_predict_tf.yaml
```

