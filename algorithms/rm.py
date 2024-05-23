"""
python rm.py \
    --model_name_or_path="mistralai/Mistral-7B-v0.1" \
    --output_dir="./rm" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""
import warnings

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
import pandas as pd
from datasets import DatasetDict, Dataset
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, model_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    reward_config.save_strategy="steps"
    reward_config.save_steps=1

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )
    model.config.pad_token_id = model.config.eos_token_id

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    df = pd.read_csv(reward_config.train_file, sep='\t', header=0)

    def create_dataset(df: pd.DataFrame):
        return Dataset.from_dict(
            {
                "translation": [
                    {'chosen':tr, 'rejected':ref}
                    for ref, tr, in zip(df["toxic-en"], df["neutral-en"])
                ]
            }
        )
    raw_datasets = DatasetDict(
        train=create_dataset(train_data),
        validation=create_dataset(val_data),
        test=create_dataset(test_data)
    )

    def preprocess_function(examples):
        
        examples = examples["translation"]
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        # for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        for item in examples: #["chosen"], examples["rejected"]):   
            tokenized_chosen = tokenizer(item["chosen"])
            tokenized_rejected = tokenizer(item["rejected"])

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model("./rm_falcon")
