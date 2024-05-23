import argparse
import csv

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.import_utils import is_npu_available, is_xpu_available

print("loaded imports")
toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")
print("loaded metric")
ds = load_dataset("/home/sslashinin/kovakimyan/diploma/dataset/OxAISH-AL-LLM___wiki_toxic/", split="test")
print("ds loaded")
parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--model_type", default="all", type=str, help="Relative path to the source model folder")
parser.add_argument("--output_file", default="toxicity.csv", type=str, help="Relative path to the source model folder")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--num_samples", default=400, type=int, help="Number of samples")
parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
parser.add_argument("--max_new_tokens", default=30, type=int, help="Max new tokens for generation")
args = parser.parse_args()

MODELS_TO_TEST = [args.model_type]
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
max_new_tokens = args.max_new_tokens
context_length = args.context_length
if is_xpu_available():
    device = torch.xpu.current_device()
elif is_npu_available():
    device = torch.npu.current_device()
else:
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

file = open(f"{output_file}", "w", newline="")
writer = csv.writer(file)
writer.writerow(["model_id", "mean_toxicity", "std_toxicity"])


for model_id in tqdm(MODELS_TO_TEST):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_texts = []

    for i, example in enumerate(ds):
        torch.manual_seed(42)

        input_text = example["comment_text"]
        input_texts.append(input_text[:2000])

        if i > NUM_SAMPLES:
            break

        if (i + 1) % BATCH_SIZE == 0:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            inputs.input_ids = inputs.input_ids[:context_length]
            inputs.attention_mask = inputs.attention_mask[:context_length]
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [
                generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
            ]
            toxicity_score = toxicity.compute(predictions=generated_texts)
            input_texts = []

            if model_id not in toxicities:
                toxicities[model_id] = []
            toxicities[model_id].extend(toxicity_score["toxicity"])

    # last batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=30)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]
    toxicity_score = toxicity.compute(predictions=generated_texts)
    toxicities[model_id].extend(toxicity_score["toxicity"])

    mean = np.mean(toxicities[model_id])
    std = np.std(toxicities[model_id])

    writer.writerow([model_id, mean, std])

    print(f"Model: {model_id} - Mean: {mean} - Std: {std}")

    model = None
    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

file.close()
