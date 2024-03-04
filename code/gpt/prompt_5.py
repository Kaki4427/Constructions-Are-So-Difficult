""""5. premise: \n hypothesis. \n Classify as yes, no, or uncertain. (Statement + Uncertain version)
"""
from chatgpt import get_model_response
import csv
import json
import re
import pandas as pd

def replace_parentheses(text):
    # Use regular expressions to perform the replacements
    text = re.sub(r'\(so\)', 'so', text)
    text = re.sub(r'\(that\)', 'that', text)
    return text

def add_not(sentence, adj):
    # Use re.sub with a limit parameter of 1 to remove only the first occurrence
    text = re.sub(r'so ' + adj, 'not so ' + adj, sentence)
    return text

def split_sentence_at_first_that_after_adj(sentence, adj):
    # Create a regular expression pattern to find "that" after the specified word
    pattern = re.compile(fr'(?<={re.escape(adj)}).*?(?=\s+that)', re.IGNORECASE)

    # Find the match
    match = pattern.search(sentence)

    if match:
        # Split the sentence into two parts using the match
        part1 = sentence[:match.end()]
        part2 = sentence[match.end():]
        return part1.strip(), part2.strip()[5:]
    else:
        print(sentence)

def generate_prompt_5(data, ignore_cache=False, model="gpt-3.5-turbo"):
    # creat the prompt
    user_prompt_list = []
    for index, row in data.iterrows():
        premise = add_not(replace_parentheses(row["sentence"]), row["adjective"])
        part1, hypothesis = split_sentence_at_first_that_after_adj(premise, row["adjective"])
        user_prompt = f"premise: {premise} \n hypothesis: {hypothesis.capitalize()} \n Classify as entailment, no entailment, or contradiction."
        user_prompt_list.append(user_prompt)

    # get the answers from ChatGPT
    system_prompt = ""
    outputs = get_model_response(user_prompt_list, system_prompt, ignore_cache=ignore_cache, model=model)

    words_to_check = ["entailment", "no_entail", "contradiction"]
    word_patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in words_to_check]

    new_outputs = []
    for i, output in enumerate(outputs):
        found_words = []
        while len(found_words) != 1:
            output_test = output.lower().replace("no entailment", "no_entail")
            for word, pattern in zip(words_to_check, word_patterns):
                if pattern.search(output_test):
                    found_words.append(word)
            if len(found_words) == 1:
                answer = found_words[0]
            else:
                output = get_model_response([user_prompt_list[i]], system_prompt, ignore_cache=True, model=model)[0]
                found_words = []
        new_outputs.append((output, answer))

    outputs, answers = zip(*new_outputs)

    outputs = list(outputs)
    answers = list(answers)

    # Initialize an empty list to store JSON objects
    json_list = []
    labels = [row["label"] for index, row in data.iterrows()]
    gold_dict = {"causal excess": "contradiction", "causal": "entailment", "non-causal": "no_entail", "only causal excess": "contradiction"}
    golds = list(map(lambda key: gold_dict.get(key, key), labels))
    for test, output, answer, label, gold in zip(user_prompt_list, outputs, answers,
                                                 labels, golds):
        result = {"label": label, "test": test, "output": output, "answer": answer, "gold": gold}
        json_list.append(result)
    json_list.append(gold_dict)

    if model == "gpt-4-0613":
        fold = "./gpt4"
    elif model == "gpt-3.5-turbo":
        fold = "./gpt3_5"
    # Convert the list of dictionaries to a JSON object
    json_data = json.dumps(json_list)
    file_path = f"{fold}/result_5.json"

    # Write the JSON data to the file
    with open(file_path, "w") as json_file:
        json_file.write(json_data)

    print(f"JSON data saved to {file_path}")

if __name__ == "__main__":
    data = pd.read_excel("combi_3.xlsx")
    generate_prompt_5(data, ignore_cache=True, model="gpt-3.5-turbo")
    generate_prompt_5(data, ignore_cache=True, model="gpt-4-0613")
