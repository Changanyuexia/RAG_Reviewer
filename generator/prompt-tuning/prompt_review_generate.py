import pandas as pd
import csv
import json
import os
import argparse

def read_jsonl_to_dataframe(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            entry = json.loads(line)
            code = entry.get('function', '')
            review = entry.get('text', '')
            data.append([code, review])
    return pd.DataFrame(data, columns=['code', 'review'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate JSONL file from CSV.')
    parser.add_argument('--Top_k', type=int, help='Top k review index to use')
    parser.add_argument('--test_top10_path', type=str, help='Path to test_top10 CSV file')
    parser.add_argument('--input_file', type=str, help='Output JSONL file path')
    args = parser.parse_args()

    # Read test_top10 data
    test_top10 = pd.read_csv(args.test_top10_path)
    reviews = []
    prompts = []

    for index, row in test_top10.iterrows():
        code = row['code']
        review = row['review']
        top1 = row['top_1']
        top1_prompt = f"You are a code reviewer. Could you please write a review for following code snippet: '{code}'? Keep the format and content of your review align with following example review: '{top1}'. Limit your review to 1 or 2 sentences, and do not introduce any code details."
        _prompt = f"You are a code reviewer. Could you please provide a concise review for following code snippet: '{code}'? Limit your review to 1 or 2 sentences, do not introduce any code details."

        new_prompt = _prompt if args.Top_k == 0 else top1_prompt
        prompts.append(new_prompt)
        reviews.append(review)

    # Ensure the output directory exists
    if not os.path.exists('input'):
        os.makedirs('input')

    # Write to JSONL file
    with open(args.input_file, 'w') as file:
        for prompt, review in zip(prompts, reviews):
            input_data = {'prompt': prompt, 'review': review}
            file.write(json.dumps(input_data) + '\n')

