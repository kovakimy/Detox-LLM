# peft:
"""
python examples/scripts/sft.py \
    --model_name_or_path="tiiuae/falcon-7b-instruct" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
import pandas as pd
from datasets import DatasetDict, Dataset
tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    
    training_args.save_strategy="steps"
    training_args.save_steps=5
    
    train_file = "/home/sslashinin/kovakimyan/diploma/experiments/detoxification_results.json"
    df = pd.pandas.read_json(train_file)
    def create_dataset(df: pd.DataFrame):
        return Dataset.from_dict(
            {
                "corpora": [
                    "Toxic example: " + tox[0] + "\nNon-toxic example: " + not_tox[0]
                   # "\nNon-toxic example: " + not_tox[0]
                    for tox, not_tox, in zip(df["toxic"], df["not_toxic"])
                ]
            }
        )
    train_ratio = 0.9
    val_ratio = 0.1
    test_ratio = 0.
    train_data = df.sample(frac=train_ratio, random_state=42)
    val_data = df.drop(train_data.index).sample(frac=val_ratio / (val_ratio + test_ratio), random_state=42)
    test_data = df.drop(train_data.index).drop(val_data.index)
    raw_datasets = DatasetDict(
        train=create_dataset(train_data),
        validation=create_dataset(val_data),
        test=create_dataset(test_data)
    )
    
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token


    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']

    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            dataset_text_field = "corpora",
        )

    trainer.train()

    with save_context:
        trainer.save_model("./sft_mistral_paralccorp")
