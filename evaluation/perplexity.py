import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = test["text"]
input_texts = [s for s in texts if s!='']
perplexity = evaluate.load("perplexity", module_type="metric")


names = ['sft_Falcon_7B_non_toxic',
 'ppo_Falcon_7B',
 'ppo_Falcon_7B_bert',
 'ppo_Mistral_7B_bert',
 'ppo_Mistral_7B_deberta',
 'falcon_7b',
 'sft_Mistral_7B_non_toxic',
 'sft_Mistral_7B_parallel_corpora',
 'ppo_Falcon_7B_roberta',
 'Mistral_7B',
 'sft_Falcon_7B_parallel_corpora',
 'ppo_Falcon_7B_deberta',
 'ppo_Mistral_7B_roberta',
 'ppo_Mistral_7B']

paths = ['/home/sslashinin/kovakimyan/diploma/experiments/models/sft_Falcon_7B_non_toxic',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Falcon_7B',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Falcon_7B_bert',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Mistral_7B_bert',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Mistral_7B_deberta',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/falcon_7b',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/sft_Mistral_7B_non_toxic',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/sft_Mistral_7B_parallel_corpora',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Falcon_7B_roberta',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/Mistral_7B',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/sft_Falcon_7B_parallel_corpora',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Falcon_7B_deberta',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Mistral_7B_roberta',
 '/home/sslashinin/kovakimyan/diploma/experiments/models/ppo_Mistral_7B']


for i in range(len(paths)):
    print(names[i])
    results = perplexity.compute(model_id=paths[i],
                                 add_start_token=False,
                                 predictions=input_texts)
    print(results["mean_perplexity"], "\n")
