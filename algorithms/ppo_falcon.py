"""
python ppo.py \
    --log_with=wandb \
    --evaluation_strategy="steps" \
    --max_length=256 \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 \
    --per_device_train_batch_size=32 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available


tqdm.pandas()


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    
    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

ppo_config.batch_size = 16
ppo_config.backward_batch_size = 16
ppo_config.mini_batch_size = 4
ppo_config.ppo_epochs = 1
print(ppo_config)

sent_kwargs = {"batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

def build_dataset(
    config, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=10
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")
    print("loaded_dataset")
    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > 0.3

    ds = ds.filter(filter_fn, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds

dataset = build_dataset(ppo_config, "allenai/real-toxicity-prompts", 30, 40)
print("ds build")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

set_seed(ppo_config.seed)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code, device_map = "auto")

device_map = "auto"
print("ref model build")
model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

print("model build")
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

tokenizer.pad_token_id = tokenizer.eos_token_id
print("tokenizer build")
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
print("ppo_trainer build ")

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = ppo_config.reward_model.split(":")
task = "text-classification"
# reward_model_name="/home/sslashinin/kovakimyan/diploma/trl/examples/rm/checkpoint-10/"
reward_model_name="/home/sslashinin/kovakimyan/diploma/trl/examples/rm_falcon/checkpoint-10/"
print("\n\n", task, reward_model_name, "\n\n")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        pipe = pipeline(task, model=reward_model_name, device_map="auto")
else:
    pipe = pipeline(task, model=reward_model_name, device_map="auto")

if pipe.tokenizer.pad_token_id is None:
    pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if pipe.model.config.pad_token_id is None:
    pipe.model.config.pad_token_id = tokenizer.pad_token_id

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}
output_min_length = 20
output_max_length = 200
output_length_sampler = LengthSampler(output_min_length, output_max_length)
epochh = 0
for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len
    
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)
    # print("\n\n", batch["response"], "\n\n")
    # print("\n\n", batch["ref_response"], "\n\n")

    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = pipe(texts, **sent_kwargs)
    # print("\n\n", pipe_outputs, "\n\n")
    rewards = [torch.tensor(output["score"]) for output in pipe_outputs]
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards
    print("\n\n", rewards, "\n\n")
    print("\n\n", ref_rewards, "\n\n")

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    # Save model every 100 epochs
    if _epoch % 5 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(f"./ppo_rm_falcon/{epochh}")
            epochh = epochh + 5