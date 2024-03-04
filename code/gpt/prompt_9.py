from chatgpt import get_model_response
import json
import re
import pandas as pd

def replace_parentheses(text):
    # Use regular expressions to perform the replacements
    text = re.sub(r'\(so\)', 'so', text)
    text = re.sub(r'\(that\)', 'that', text)
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

def generate_prompt_9(data, ignore_cache=False, model="gpt-3.5-turbo"):
    user_prompt_list = []
    for index, row in data.iterrows():
        premise = replace_parentheses(row["sentence"])
        part1, part2 = split_sentence_at_first_that_after_adj(premise, row["adjective"])
        user_prompt = f"{premise} \nPart1: {part1} \nPart2: {part2.capitalize()} \nCan we infer that Part1 is the cause of Part2? \nAnswer with yes, no or uncertain."
        user_prompt_list.append(user_prompt)

    # get the answers from ChatGPT
    system_prompt = ""
    outputs = get_model_response(user_prompt_list, system_prompt, ignore_cache=ignore_cache, model=model)

    words_to_check = ["yes", "uncertain", "no"]
    word_patterns = [re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE) for word in words_to_check]

    new_outputs = []
    for i, output in enumerate(outputs):
        found_words = []
        while len(found_words) != 1:
            for word, pattern in zip(words_to_check, word_patterns):
                if pattern.search(output):
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
    gold_dict = {"causal excess": "yes", "causal": "no", "non-causal": "no", "only causal excess": "yes"}
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
    file_path = f"{fold}/result_9.json"

    # Write the JSON data to the file
    with open(file_path, "w") as json_file:
        json_file.write(json_data)

    print(f"JSON data saved to {file_path}")


if __name__ == "__main__":
    data = pd.read_excel("combi_3.xlsx")
    generate_prompt_9(data, ignore_cache=True, model="gpt-3.5-turbo")
    generate_prompt_9(data, ignore_cache=True, model="gpt-4-0613")
