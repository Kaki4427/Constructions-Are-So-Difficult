# !pip install transformers
# !pip install accelerate
# !pip install optimum
# !pip install auto-gptq

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os

model_name = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

test_path = "./test/"
answer_path = "llama2/"

if answer_path is not None:
    os.makedirs(answer_path, exist_ok=True)
    files = os.listdir(answer_path)
    existing_sentences_set = set(files)

B_INST, E_INST = "<s>[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def create_prompt(instruction, b_inst=B_INST, e_inst=E_INST): # missing system_prompt=DEFAULT_SYSTEM_PROMPT
    return b_inst + instruction + e_inst

def format_output(raw_out, prompt, tokenizer):
    # remove input prompt
    out = raw_out.replace(prompt, '')

    # remove BOS and EOS tokens
    out = out.replace(tokenizer.bos_token, '')
    out = out.replace(tokenizer.eos_token, '')

    # remove any empty spaces
    out = out.strip()
    return out

def generate(instruction, do_return=False, model=model, tokenizer=tokenizer): # missing sys_prompt=DEFAULT_SYSTEM_PROMPT
    prompt = create_prompt(instruction=instruction) # missing system_prompt=sys_prompt
    input_idx = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
    out_idx = model.generate(input_idx, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    out = tokenizer.decode(out_idx[0])

    clean_out = format_output(out, prompt=prompt, tokenizer=tokenizer)

    if do_return:
        return clean_out
    print(clean_out)

for i in range(1, 17):
    test_file_name = f"{test_path}result_{i}.json"
    answer_file_name =  f"{answer_path}answer_{i}.json"

    with open(test_file_name, "r") as file:
        data = json.load(file)

    print(f"processing {test_file_name}!")

    # gold labels of this prompt
    gold = set([item['gold'] for item in data[:-1]])
    gold.add("no entailment")

    tests = [item["test"] for item in data[:-1]]

    results = []

    for instruction in tqdm(tests, desc="Processing"):
        result = dict()
        output = generate(instruction, do_return=True)

        # check the output: it there any lable in the result
        flag = any(word.lower() in output.lower() for word in gold)
        while not flag:
            output = generate(instruction, do_return=True)
            flag = any(word.lower() in output.lower() for word in gold)

        result["input"] = instruction
        result["output"] = output
        results.append(result)

    with open(answer_file_name, "w") as file:
        json.dump(results, file)
    print(f"{answer_file_name} is completed!")
