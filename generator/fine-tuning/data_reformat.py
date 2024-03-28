#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RAG-CodeReviewer 
@File ：data_reformat.py
@Date ：2023/12/20 22:43 
'''

import os
import csv
import argparse
import pandas as pd
import json

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import os
import argparse

def create_arg_parser():
    current_directory = os.getcwd()
    train_default_path = os.path.join(current_directory, "../../retriever/gpm_retrieval/cc/train_top10.csv")
    test_default_path = os.path.join(current_directory, "../../retriever/gpm_retrieval/cc/test_top10.csv")
    save_path_default = os.path.join(current_directory, "prompt/cc/")

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", '--train', default=train_default_path,
                        type=str, help="path to train file")
    parser.add_argument("-test", '--test', default=test_default_path,
                        type=str, help="path to test file")
    parser.add_argument("-save", "--save_path", default=save_path_default, type=str,
                        help="Path where to save the output.")
    args = parser.parse_args()
    return args

def read_jsonl_to_dataframe(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            entry = json.loads(line)
            code = entry.get('function', '')
            review = entry.get('text', '')
            data.append([code, review])
    return pd.DataFrame(data, columns=['code', 'review'])


if __name__ == '__main__':
    # load the arguments
    args = create_arg_parser()
    train_path = args.train
    test_path = args.test
    save_path = args.save_path
    print(save_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(len(train))
    # fine tuning
    train_prompt = []
    train_prompt_top1 = []
    for index, row in train.iterrows():
        code = row['code']
        review = row['review']
        top1_review = row['top_1']
        format = {
            "instruction": "Please review the following code and provide a one-line code review comment in English:",
            "input": f"{code}",
            "output": f"{review}"
        }
        format = {
            "instruction": f"You are a code reviewer. Could you please provide a concise review for following code snippet? Limit your review to 1 or 2 sentences, do not introduce any code details.",
            "input": f"{code}",
            "output": f"{review}",
        }
        train_prompt.append(format)

        format = {
            "instruction": f"You are a code reviewer. Could you please provide a concise review for following code snippet? Keep the format and content of your review similar to given example review: '{top1_review}'. Limit your review to 1 or 2 sentences, do not introduce any code details.",
            "input": f"{code}",
            "output": f"{review}",
        }
        train_prompt_top1.append(format)

    print(len(train_prompt_top1))
    file_path = save_path + "code_review_train.json"
    with open(file_path, 'w') as json_file:
        json.dump(train_prompt, json_file)


    file_path = save_path + "code_review_train_top1.json"
    with open(file_path, 'w') as json_file:
        json.dump(train_prompt_top1, json_file)
    
    print(file_path)

    # test prompt
    test_prompt = []
    test_prompt_top1 = []
    for index, row in test.iterrows(): 
        # choose top1 item
        code = row["code"]
        review = row["review"]
        top1_review = row["top_1"]

        # vanilla prompt
        format = {
            "instruction": "Please review the following code and provide a one-line code review comment in English:",
            "input": f"{code}",
            "output": f"{review}"
        }

        format = {
            "instruction": f"You are a code reviewer. Could you please provide a concise review for following code snippet? Limit your review to 1 or 2 sentences, do not introduce any code details.",
            "input": f"{code}",
            "output": f"{review}",
        }
        test_prompt.append(format)

        # prompt with top1
        # top1_content = train.iloc[top1]
        # top1_code = top1_content["code"]
        # top1_review = top1_content["review"]
        # format = {
        #   "instruction": f"Here is an example of a piece of code: {top1_code} and its review: {top1_review}. "
        #             f"Please review the following code and provide a constructive code review comment:",
        #   "input": f"{code}",
        #   "output": f"{review}",
        # }
        # format = {
        #    "instruction": "You are a code reviewer, referring to the similar code-comment pair to give a short review with the same length as the reference review in English.",
        #    "reference code": f"{top1_code}",
        #    "reference review": f"{top1_review}",
        #    "input": f"{code}",
        #    "output": f"{review}",
        # }
        format = {
            "instruction": f"You are a code reviewer. Could you please provide a concise review for following code snippet? Keep the format and content of your review similar to given example review: '{top1_review}'. Limit your review to 1 or 2 sentences, do not introduce any code details.",
            "input": f"{code}",
            "output": f"{review}",
        }
        test_prompt_top1.append(format)
    file_path = save_path + "code_review_test.json"
    print(file_path)
    with open(file_path, 'w') as json_file:
        json.dump(test_prompt, json_file)

    file_path = save_path + "code_review_test_top1.json"
    with open(file_path, 'w') as json_file:
        json.dump(test_prompt_top1, json_file)
    
