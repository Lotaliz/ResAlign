import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "../models/llama3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
data_path = "../datasets/SafetyBench/test_zh.json"
with open(data_path, 'r') as f:
  dataset = json.load(f)

output_data = {}

for idx, entry in enumerate(dataset):
  question = entry['question']
  options = entry['options']
  input_text = f"Question: {question}\nOptions: {', '.join(options)}\nAnswer:"
  
  inputs = tokenizer(input_text, return_tensors='pt').to(device)
  with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
  answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

  answer = answer.split("Answer:")[-1].strip()

  selected_choice_idx = -1
  for i, choice in enumerate(options):
    if choice.lower() in answer.lower():
      selected_choice_idx = i
      break

  output_data[str(idx)] = selected_choice_idx

output_json_file = 'output_predictions.json'
with open(output_json_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Evaluated {len(dataset)} questions and saved the results to '{output_json_file}'")
