from transformers import AutoTokenizer

# Load the LLaMA tokenizer (assuming the model is available in Hugging Face hub)
tokenizer = AutoTokenizer.from_pretrained("../models/deepseek-r1-distill-llama-8b")

token = tokenizer.convert_tokens_to_ids('B')
# token = tokenizer.convert_ids_to_tokens(32)

print(token)
