#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

source ~/.bashrc
conda activate generate
Top_k=1
Dataset="tf"
python prompt_review_generate.py \
    --Top_k $Top_k \
    --test_top10_path ../../retriever/gpm_retrieval/${Dataset}/test_top10.csv \
    --input_file input/${Dataset}/model_input_top_${Top_k}.jsonl 
