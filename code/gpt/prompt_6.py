from chatgpt import get_model_response
import json
import re
import pandas as pd
import spacy

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
def generate_question(clause):
    # Load the spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    # Input sentence
    input_sentence = ("He worked at MIT . ")
    input_sentence = input_sentence[0].lower() + input_sentence.strip()[1:-1]

    # Process the input sentence using spaCy
    doc = nlp(input_sentence)

    # Initialize an empty list for the modified words
    modified_words = ["Did"]

    # Iterate through the tokens in the processed sentence
    for token in doc:
        # Check if the token is a verb in past tense
        if token.pos_ == "VERB" and token.tag_ == "VBD":
            # Replace the past tense verb with its base form (lemma_)
            modified_words.append(token.lemma_)
        else:
            modified_words.append(token.text)

    modified_words.append("?")

    # Join the modified words back into a sentence
    output_sentence = " ".join(modified_words)

    return output_sentence


def generate_prompt_6(user_prompt_list, data, ignore_cache=False, model="gpt-3.5-turbo"):

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
    gold_dict = {"causal excess": "no", "causal": "yes", "non-causal": "uncertain", "only causal excess": "no"}
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
    file_path = f"{fold}/result_6.json"

    # Write the JSON data to the file
    with open(file_path, "w") as json_file:
        json_file.write(json_data)

    print(f"JSON data saved to {file_path}")

if __name__ == "__main__":
    with open("prompt6.json", "r") as file:
        user_prompt_list = json.load(file)
    data = pd.read_excel("combi_3.xlsx")
    generate_prompt_6(user_prompt_list, data, ignore_cache=True, model="gpt-3.5-turbo")
    generate_prompt_6(user_prompt_list, data, ignore_cache=True, model="gpt-4-0613")
