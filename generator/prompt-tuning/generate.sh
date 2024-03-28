#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

source ~/.bashrc
conda activate generate
Topk=1
Dataset="tf"
# Run the python script with arguments
python generate.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --input_file input/${Dataset}/model_input_top_${Topk}.jsonl \
    --output_file output/${Dataset}/generate_${Topk}_m7b.jsonl \
    --evaluation_file output/${Dataset}/eval_${Topk}_m7b.jsonl
