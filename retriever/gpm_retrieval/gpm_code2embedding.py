#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RAG-CodeReviewer 
@File ：code2embedding.py
@Date ：2023/12/17 7:41 
'''
import time
from difflib import SequenceMatcher
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
    parser.add_argument("-train", '--train', default=os.path.join(path, "../../data/tf/train.csv"), type=str,
                        help="path of train file")
    parser.add_argument("-test", '--test', default=os.path.join(path, "../../data/tf/test.csv"), type=str,
                        help="path to test file")
    parser.add_argument("-val", '--val', default=os.path.join(path, "../../data/tf/val.csv"), type=str,
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

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() 

def predictionTopk(topk, similarity_matrix, train, test):
    predictions = []

    print("Processing:", topk)
    start_time = time.time()

    for index in range(len(similarity_matrix)):
        if index % 1000 == 0:
            print("Processing instance", index, "/", len(similarity_matrix))

        # get cosine similarity candidates
        index_nn = np.argpartition(similarity_matrix[index], -topk)[-topk:]
        similar_nn = []
        for idx in index_nn:
            # get gpm similarity candidates
            similar_score = similar(test.iloc[index]['code'], train.iloc[idx]['code'])
            similar_nn.append((idx, similar_score))

        similar_nn.sort(key=lambda x: x[1], reverse=True)
        topk_indices = [element[0] for element in similar_nn[:topk]]

        topk_reviews = train.iloc[topk_indices]['review'].values
        predictions.append(topk_reviews)

    print(topk, "time cost:", time.time() - start_time, "seconds")
    return predictions

def get_cosine_similarity(embeddings1, embeddings2):
    # Ensure all embeddings have the same shape
    # This step might require you to reshape or truncate your embeddings

    similarity_start_time = time.time()
    # Convert lists to numpy arrays
    arr1 = np.array(embeddings1)
    arr2 = np.array(embeddings2)

    # Check if the arrays are 2-dimensional
    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError("Both arrays need to be 2-dimensional.")

    # Compute cosine similarity
    cosine_similarity = 1 - cdist(arr2, arr1, 'cosine')
    if np.array_equal(arr1, arr2):
        np.fill_diagonal(cosine_similarity, -1)

    return cosine_similarity

def integrate_predictions(train, predictions):
    final_df = train[['code', 'review']].copy()
    
    for i in range(1, 11):
        final_df[f'top_{i}'] = [pred[i-1] for pred in predictions]
    
    return final_df


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)

    def get_embeddings(data_snippets):
        embeddings = []
        for batch in tqdm(batch_generator(data_snippets, batch_size=args.batch_size)):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=args.max_length).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    # 获取嵌入
    train_embedding = get_embeddings(train["code"])
    test_embedding = get_embeddings(test["code"])
    val_embedding = get_embeddings(val["code"])

    # 计算相似度并保存结果
    cosine_similarity = get_cosine_similarity(train_embedding, train_embedding)
    topk_predictions = predictionTopk(10, cosine_similarity, train, train)
    final_df = integrate_predictions(train, topk_predictions)
    final_df.to_csv("../../data/tf/train_top10.csv", index=False)
