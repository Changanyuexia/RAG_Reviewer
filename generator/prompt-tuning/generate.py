import torch
from typing import List, Dict
from transformers import AutoTokenizer, pipeline
import json
from tqdm import tqdm
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import argparse

class EvaluationMetrics:
    """
    A class to evaluate the performance of a sequence-to-sequence model using BLEU and ROUGE metrics.
    """
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        perfect_predictions_count = 0  # Initialize the counter for perfect predictions

        for pred, ref in zip(predictions, references):
            # Check for a perfect prediction
            if " ".join(pred.split()) == " ".join(ref.split()):
                perfect_predictions_count += 1

            hypothesis = pred.split()
            reference = ref.split()

            if len(hypothesis) == 0 or len(reference) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(ref)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        # Calculate the mean for the metrics
        averaged_scores = {k: np.mean(v) for k, v in score_dict.items()}
        # Add the count of perfect predictions to the results
        averaged_scores['pp'] = perfect_predictions_count

        return averaged_scores


def load_data(input_file):
    prompts = []
    reviews = []
    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['prompt'])
            reviews.append(data['review'])
    return prompts, reviews
def load_data(input_file):
    prompts = []
    reviews = []
    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['prompt'])
            reviews.append(data['review'])
    return prompts, reviews

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text and evaluate the model.")
    parser.add_argument("--model", type=str, required=True, help="Model identifier on Huggingface.co")
    parser.add_argument("--input_file", type=str, required=True, help="Input file path for prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path for generated texts.")
    parser.add_argument("--evaluation_file", type=str, required=True, help="Evaluation file path for metrics.")
    args = parser.parse_args()

    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(model)
    text_generator = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    prompts, reviews = load_data(args.input_file)

    results = []
    for i in tqdm(range(0, len(prompts), 5), desc="Generating responses"):
        batch_prompts = prompts[i:i + 5]
        batch_results = text_generator(
            batch_prompts,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            min_new_tokens=30,
            max_new_tokens=120
        )
        results.extend([res[0]['generated_text'] for res in batch_results])
    cleaned_results = []
    for result, prompt in zip(results, prompts):
        cleaned_result = result.replace(prompt, "").strip()
        cleaned_results.append(cleaned_result)

    with open(args.output_file, 'w', encoding='utf-8') as file:
        for result, review, prompt in zip(cleaned_results, reviews, prompts):
            data = {"result": result, "review": review, "prompt": prompt}
            file.write(json.dumps(data) + '\n')

    evaluator = EvaluationMetrics()
    evaluation_results = evaluator.compute_metrics(cleaned_results, reviews)
    with open(args.evaluation_file, 'w', encoding='utf-8') as file:
        json.dump(evaluation_results, file) 
