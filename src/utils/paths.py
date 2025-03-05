MODEL_PATH = {
  'Llama': 'llama3.2-1B-Instruct',
  'Qwen': 'Qwen2.5-1.5B-Instruct',
  'Llama_f': 'llama3.2-1B-Instruct-finetuned',
  'Qwen_f': 'Qwen2.5-1.5B-Instruct-finetuned'
}

DATASET_PATH = {
  'SafetyBench_zh': 'SafetyBench/test_zh.json',
  'SafetyBench_en': 'SafetyBench/test_en.json',
  'CQIA_Tradition': 'COIG-CQIA/chinese_traditional',
}

OUTPUT_PATH = {
  'Llama_SB_zh': 'SB/llama/predictions_zh.json',
  'Llama_SB_en': 'SB/llama/predictions_en.json',
  'Qwen_SB_zh': 'SB/qwen/predictions_zh.json',
  'Qwen_SB_en': 'SB/qwen/predictions_en.json',
  'Llamaf_SB_zh': 'SB/llama/finetuned_pre_zh.json',
  'Llamaf_SB_en': 'SB/llama/finetuned_pre_en.json',
  'Qwenf_SB_zh': 'SB/qwen/finetuned_pre_zh.json',
  'Qwenf_SB_en': 'SB/qwen/finetuned_pre_en.json',
}

ANOMALY_PATH = {
  'Llama_SB_zh': 'SB/llama/anomalies_zh.json',
  'Llama_SB_en': 'SB/llama/anomalies_en.json',
  'Qwen_SB_zh': 'SB/qwen/anomalies_zh.json',
  'Qwen_SB_en': 'SB/qwen/anomalies_en.json',
  'Llamaf_SB_zh': 'SB/llama/finetuned_anom_zh.json',
  'Llamaf_SB_en': 'SB/llama/finetuned_anom_en.json',
  'Qwenf_SB_zh': 'SB/qwen/finetuned_anom_zh.json',
  'Qwenf_SB_en': 'SB/qwen/finetuned_anom_en.json',
}