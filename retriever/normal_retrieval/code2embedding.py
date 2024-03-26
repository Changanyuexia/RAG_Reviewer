#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RAG-CodeReviewer 
@File ：code2embedding.py
@Date ：2023/12/17 7:41 
'''


import os
import csv
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

from transformers import RobertaTokenizer, RobertaModel

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", '--train', default=os.path.join(path, "../data/cc/train.csv"), type=str,
                        help="path of train file")
    parser.add_argument("-test", '--test', default=os.path.join(path, "../data/cc/test.csv"), type=str,
                        help="path to test file")
    parser.add_argument("-val", '--val', default=os.path.join(path, "../data/cc/val.csv"), type=str,
                        help="path of val file")
    parser.add_argument("-embedding", '--embedding_model', default="microsoft/codebert-base", type=str,
                        help="choose one embedding_model")
    parser.add_argument("-batch", '--batch_size', default=32, type=int,
                        help="batch size")
    parser.add_argument("-max_length", '--max_length', default=256, type=int,
                        help="max length of model")
    parser.add_argument("-topk", '--topk', default=10, type=int,
                        help="top-k similar thing")
    args = parser.parse_args()
    return args


def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        # Extract the batch
        batch = data[i:i + batch_size]
        # Ensure each element in the batch is a string
        batch = [str(item) for item in batch]
        yield batch


def find_top_k_similar(embeddings1, embeddings2, other_df, k=10):
    # Ensure all embeddings have the same shape
    # This step might require you to reshape or truncate your embeddings

    # Convert lists to numpy arrays
    arr1 = np.array(embeddings1)
    arr2 = np.array(embeddings2)

    # Check if the arrays are 2-dimensional
    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError("Both arrays need to be 2-dimensional.")

    # Compute cosine similarity
    cosine_similarity = 1 - cdist(arr2, arr1, 'cosine')

    # Get the indices of top k similar embeddings for each item in arr2
    top_k_indices = np.argsort(-cosine_similarity, axis=1)[:, :k]

    #return top_k_indices
    # Fetch the reviews corresponding to the top-k indices
    top_k_reviews = []
    for indices in top_k_indices:
        reviews = other_df.iloc[indices]['review'].values
        top_k_reviews.append(reviews)
    
    return top_k_reviews



if __name__ == '__main__':
    # load the arguments
    args = create_arg_parser()
    train_path = args.train
    test_path = args.test
    val_path = args.val
    model_name = args.embedding_model
    batch_size = args.batch_size
    max_length = args.max_length
    
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                json_obj = json.loads(line)
                function = json_obj.get('function', '').replace('\n', ' ')
                text = json_obj.get('text', '')
                data.append({'code': function, 'review': text})
        return pd.DataFrame(data)

    # read csv file
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    val = pd.read_csv(val_path)
    print(f"length of train: {len(train)}\nlength of val: {len(val)}\nlength of test: {len(test)}")

    # load the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)

    ## calculate the code embedding
    def get_embeddings(data_snippets):
        embeddings = []
        for batch in tqdm(batch_generator(data_snippets, batch_size=args.batch_size)):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=args.max_length).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    train_embedding = get_embeddings(train["code"])
    test_embedding = get_embeddings(test["code"])
    val_embedding = get_embeddings(val["code"])

    def calculate_similarity_and_save(train_embedding, other_embedding, other_df, save_path):
        top_k_reviews = find_top_k_similar(train_embedding, other_embedding,other_df, 10)
        # Since we now have a list of numpy arrays containing reviews, we need to handle it accordingly
        # Let's create a DataFrame to store these reviews
        reviews_df = pd.DataFrame(top_k_reviews, columns=[f'top_{i+1}' for i in range(args.topk)])
        merged_df = pd.concat([other_df.reset_index(drop=True), reviews_df], axis=1)
        merged_df.to_csv(save_path, index=False)
    calculate_similarity_and_save(train_embedding, train_embedding, train, "cc/train_top10.csv")
    calculate_similarity_and_save(train_embedding, test_embedding, test, "cc/test_top10.csv")
    calculate_similarity_and_save(train_embedding, val_embedding, val, "cc/val_top10.csv")
