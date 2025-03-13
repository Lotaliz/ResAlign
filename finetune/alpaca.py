import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description='all')
parser.add_argument('-o', '--outdir', type=str, default='../models')
parser.add_argument('-m', '--model', type=str, default='llama3.2-1B-Instruct')
parser.add_argument('-d', '--dataset', type=str, default='alpaca')
parser.add_argument('-l', '--lang', type=str, default='en', choices=['en', 'zh'])
parser.add_argument('-om', '--online_model', action='store_true')
parser.add_argument('-od', '--online_dataset', action='store_true')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = args.model if args.online_model else f'../models/{args.model}'
language = args.lang

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset_path = args.dataset if args.online_dataset else f'../datasets/alpaca/Alpaca_data_gpt4_zh.jsonl'
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Preprocess function for tokenizing inputs and labels for causal language modeling
def preprocess_function(examples, lang='en'):
    if lang == 'en':
        questions = examples['instruction']
        inputs = examples['input']
        answers = examples['output']
    
        # Create the input prompt for the model
        prompts = [f"User: {q} {i}\nAssistant: {a}" for q, i, a in zip(questions, inputs, answers)]
    elif lang == 'zh':
        questions = examples['instruction_zh']
        inputs = examples['input_zh']
        answers = examples['output_zh']
    
        prompts = [f"{q} {i}\n答案：{a}" for q, i, a in zip(questions, inputs, answers)]
    
    model_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    
    labels = model_inputs["input_ids"].copy()  
    # labels = [seq[1:] + [tokenizer.pad_token_id] for seq in labels]  

    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()

training_args = TrainingArguments(
    output_dir=f'{args.outdir}/checkpoints',
    evaluation_strategy="steps", 
    save_strategy="steps", 
    eval_steps=2500, 
    save_steps=2500,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=100,
    fp16=False, 
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

print("Starting training.")
trainer.train()

trainer.save_model(f'{args.outdir}/{args.model}-finetuned')
