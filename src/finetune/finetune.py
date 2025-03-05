import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

from utils import paths

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = f"../models/{paths.MODEL_PATH['Qwen']}"
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
  
  inputs = ["问题: " + q + " 答案:" for q in questions]
  targets = answers
  
  model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
  labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)
  
  model_inputs['labels'] = labels['input_ids']
  return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
  output_dir=f"../models/{paths.MODEL_PATH['Qwen_f']}",
  eval_strategy="epoch",
  learning_rate=1e-5,
  per_device_train_batch_size=4,
  num_train_epochs=5,
  weight_decay=0.01,
  logging_dir="../logs",
  logging_steps=100,
  bf16=True
)

trainer = Trainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  tokenizer=tokenizer
)
print("Going to train.")
trainer.train()

model.save_pretrained(f"../models/{paths.MODEL_PATH['Qwen_f']}")
tokenizer.save_pretrained(f"../models/{paths.MODEL_PATH['Qwen_f']}")
