import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils import paths

model_path = f"../models/{paths.MODEL_PATH['Llama']}"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
data_path = f"../datasets/{paths.DATASET_PATH['SafetyBench_zh']}"
with open(data_path, 'r') as f:
  dataset = json.load(f)

output_data = {}
anomalies = {}

for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Processing questions"):
  question = entry['question']
  options = entry['options']
  input_text = f"问题: {question}\n选项: {', '.join(options)}\n回答:"
  # input_text = f"Question: {question}\nOptions: {', '.join(options)}\nAnswer:"
  inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(device)
  with torch.no_grad():
    outputs = model.generate(
      inputs['input_ids'], 
      max_length=1024, 
      num_return_sequences=1,
      attention_mask=inputs['attention_mask'], 
      pad_token_id=pad_token_id
    )
  answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

  answer = answer.split("回答:")[-1]
  # answer = answer.split("Answer:")[-1].strip()

  selected_choice_idx = -1
  if answer.strip().startswith("是"):
    selected_choice_idx = 0
  elif answer.strip().startswith("否") or answer.strip().startswith("不是"):
    selected_choice_idx = 1

  if selected_choice_idx == -1:
    anomalies[str(idx)] = {
      "question": question,
      "answer": answer
    }

  output_data[str(idx)] = selected_choice_idx

output_json_file = f"../data/{paths.OUTPUT_PATH['Llama_SB_zh']}"
with open(output_json_file, 'w') as f:
    json.dump(output_data, f, indent=4)

anomalies_json_file = f"../data/{paths.ANOMALY_PATH['Llama_SB_zh']}"
with open(anomalies_json_file, 'w', encoding='utf-8') as f:
    json.dump(anomalies, f, indent=4, ensure_ascii=False)

print(f"Evaluated {len(dataset)} questions")
