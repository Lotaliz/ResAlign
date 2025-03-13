import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import openai
from tqdm import tqdm
import os

def parse_args():
    """
    Parse command line arguments for flexible model, dataset, and output paths.
    """
    parser = argparse.ArgumentParser(description="Evaluate a local language model with harm detection using OpenAI API.")
    parser.add_argument('--model', type=str, required=True, help="Path or name of the local language model.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the input Parquet dataset.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the harm evaluation results.")
    parser.add_argument('--openai-api-key', type=str, required=True, help="Your OpenAI API key for harm evaluation.")
    return parser.parse_args()

def evaluate_harm(response_text, openai_api_key):
    """
    Evaluate the potential harm of the generated response using OpenAI API.
    """
    openai.api_key = openai_api_key

    try:
        # Request for harm evaluation using OpenAI's moderation model
        response = openai.Moderation.create(input=response_text)
        harm_score = response['results'][0]['category_scores']
        return harm_score
    except Exception as e:
        print(f"Error evaluating harm: {e}")
        return None

def generate_response(prompt, model, tokenizer):
    """
    Generate a response using the local language model based on the prompt.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the dataset
    df = pd.read_parquet(args.dataset)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to('cuda')

    # Iterate through the prompts in the dataset and evaluate the responses
    harm_scores = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt = row['prompt']
        # Generate response
        response = generate_response(prompt, model, tokenizer)
        
        # Evaluate harm of the response using OpenAI API
        harm_score = evaluate_harm(response, args.openai_api_key)
        harm_scores.append({
            'prompt': prompt,
            'response': response,
            'harm_score': harm_score
        })

    # Save the results to a DataFrame
    results_df = pd.DataFrame(harm_scores)

    # Save results to the specified output path
    results_df.to_parquet(args.output)

    print(f"Evaluation complete and results saved to {args.output}.")

if __name__ == "__main__":
    main()
