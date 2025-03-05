import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

from utils import paths

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = f"../models/{paths.MODEL_PATH['Llama']}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token 
dataset_path = f"../datasets/{paths.DATASET_PATH['CQIA_Tradition']}"

jsonl_files = [
    f"{dataset_path}/chengyu.jsonl",
    f"{dataset_path}/poem.jsonl",
    f"{dataset_path}/trad-multi-choice-40.jsonl",
    f"{dataset_path}/trad-multi-choice-100-2.jsonl",
    f"{dataset_path}/trad-multi-choice-100.jsonl",
    f"{dataset_path}/translate_classical_chinese.jsonl",
]

dataset = load_dataset("json", data_files=jsonl_files, split="train")

def preprocess_function(examples):
    questions = examples['instruction']
    answers = examples['output']
    
    inputs = [f"Question: {q} Answer:" for q in questions]
    targets = answers
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=f"../models/{paths.MODEL_PATH['Llama_f']}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir="../logs",
    logging_steps=100,
    load_best_model_at_end=True,
    bf16=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained(f"../models/{paths.MODEL_PATH['Llama_f']}")
tokenizer.save_pretrained(f"../models/{paths.MODEL_PATH['Llama_f']}")
