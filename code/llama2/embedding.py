# !pip install hashlib
# !pip install transformers
# !pip install accelerate
# !pip install optimum
# !pip install auto-gptq

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import re
import pickle
import gc
import hashlib
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import os
import sys

# Empty Torch cache
torch.cuda.empty_cache()
gc.collect()

model_name = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf"
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=512)
print("Loading weights...", file=sys.stderr)
llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=4,
    trust_remote_code=True,
    )
print('Done.', file=sys.stderr)

# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

test_path = "./test/"
answer_path = "./answer/"

B_INST, E_INST = "<s>[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def create_prompts(instructions, b_inst=B_INST, e_inst=E_INST):
    return [i if i.startswith("<s>[INST]") else b_inst + i + e_inst for i in instructions]


def generate(instructions, labels, words_to_check, gold_dict, cola=False):

    prompts = create_prompts(instructions)

    results = []
    rerun_ins = [] # save the prompts need to rerun
    rerun_labels = []

    outputs = llm.generate(prompts, sampling_params)

    word_patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in words_to_check]

    for output, label in zip(outputs, labels):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        to_check = generated_text.replace("no entailment", "no_entail")

        if cola:
            found_words = []
            parts = generated_text.split("9. Sentence:")
            if len(parts) >= 2:
                text = parts[1].strip()
                ans = text.split("Answer:")
                if len(ans) >= 2:
                    answer = ans[1].strip()
                    for word, pattern in zip(words_to_check, word_patterns):
                        if pattern.search(answer):
                            found_words.append(word)
                    if found_words:
                        result = dict()
                        result["label"] = label
                        result["test"] = prompt
                        result["output"] = generated_text
                        result["found_words"] = found_words
                        result["answer"] = found_words[0]
                        result["gold"] = gold_dict.get(label, label)
                        results.append(result)
            else:
                rerun_ins.append(prompt)
                rerun_labels.append(label)
        else:
            found_words = []
            for word, pattern in zip(words_to_check, word_patterns):
                if pattern.search(to_check):
                    found_words.append(word)
            if found_words:
                result = dict()
                result["label"] = label
                result["test"] = prompt
                result["output"] = generated_text
                result["found_words"] = found_words
                result["answer"] = found_words[0]
                result["gold"] = gold_dict.get(label, label)
                results.append(result)
            else:
                rerun_ins.append(prompt)
                rerun_labels.append(label)
    if rerun_ins:
        results += generate(rerun_ins, rerun_labels, words_to_check, gold_dict, cola)
    return results


if answer_path is not None:
    os.makedirs(answer_path, exist_ok=True)
    files = os.listdir(answer_path)
    existing_sentences_set = set(files)

numbers = [i for i in range(1, 11)] + [16]
cola = [i for i in range(11, 16)]

for i in range(1, 17):
    test_file_name = f"{test_path}result_{i}.json"
    answer_file_name =  f"{answer_path}answer_{i}.json"

    with open(test_file_name, "r") as file:
        data = json.load(file)

    gold_dict = data.pop()

    print(f"processing {test_file_name}!")

    # words_to_check of this prompt
    words_to_check = set([item['gold'] for item in data])

    prompts = [item["test"] for item in data]
    labels = [item["label"] for item in data]

    if i in numbers:
        results = generate(prompts, labels, words_to_check, gold_dict)
    elif i in cola:
        results = generate(prompts, labels, words_to_check, gold_dict, cola=True)

    with open(answer_file_name, "w") as file:
        json.dump(results, file)
    print(f"{answer_file_name} is completed!")
    
file_name = "./combi_4.xlsx"
df = pd.read_excel(file_name)

def replace_parentheses(text):
    # Use regular expressions to perform the replacements
    text = re.sub(r'\(so\)', 'so', text)
    text = re.sub(r'\(that\)', 'that', text)
    return text

sentences = [replace_parentheses(s) for s in df["sentence"]]

def cache_filename(sentence):
    return hashlib.md5(sentence.encode('utf-8')).hexdigest() + '.pkl'

model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

cache_path = "./cache/"

existing_sentences_set = set()

if cache_path is not None:
    os.makedirs(cache_path, exist_ok=True)
    files = os.listdir(cache_path)
    existing_sentences_set = set(files)

new_sentences_list = [sentence for sentence in sentences if cache_filename(sentence) not in existing_sentences_set]

print(f'Getting embeddings for {len(new_sentences_list)} sentences...')

for sentence in new_sentences_list:
    tokenized = tokenizer.encode_plus(sentence, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokenized, output_hidden_states=True)
        last_layer_hidden_state = output.hidden_states[-1]
        file_path = os.path.join(cache_path, cache_filename(sentence))
        with open(file_path, 'wb') as f:
            pickle.dump(last_layer_hidden_state.squeeze().detach().numpy(), f)
