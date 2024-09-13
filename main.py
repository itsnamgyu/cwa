import json
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

HF_TOKEN = "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"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "google/gemma-2-2b-it"
# model_name = "google/gemma-2-2b"
# model_name = "meta-llama/Meta-Llama-3.1-8B"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

hf_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, device_map="sequential")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = HFLM(hf_model, batch_size=1)

results = evaluator.simple_evaluate(model, tasks=["arc_challenge", "arc_easy"], device="cuda:0")
with open("results.txt", "w") as f:
    f.write(str(results))
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)