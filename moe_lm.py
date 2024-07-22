import json
import random
import lm_utils
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help = "model name")
    parser.add_argument("-i", "--input", help = "input file name")
    parser.add_argument("-t", "--type", help = "generate or probability")
    parser.add_argument("-o", "--portion", default = 1, help = "portion of the data to use")

    args = parser.parse_args()
    model_name = args.model
    input_file = args.input
    type = args.type
    portion = float(args.portion)

    if type == "generate":
        lm_utils.llm_init(model_name, probs = False)
    elif type == "probability":
        lm_utils.llm_init(model_name, probs = True)

    # for comments loading
    temp_input_file = input_file
    if "_small" in input_file:
        temp_input_file = input_file.replace("_small", "")

    comment_pool = {}
    for domain in ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                   "mistral-reddit_center", "mistral-reddit_left", "mistral-reddit_right"]:
        comments = json.load(open("community_lm_msgs/" + temp_input_file + "_" + domain + ".json", 'r'))
        for key in comments.keys():
            if key not in comment_pool:
                comment_pool[key] = [comments[key]]
            else:
                comment_pool[key].append(comments[key])

    data = None
    with open("input/" + input_file + ".json", 'r') as f:
        data = json.load(f)

    ind_to_option_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    # smaller data size, in case it is too slow
    if portion < 1:
        data = data[:int(len(data) * portion)]
    
    for i in tqdm(range(len(data))):
        if type == "generate":
            prompt = data[i]["input"]

            chosen_expert = None
            expert_selection_prompt = data[i]["input"] + "\n\nPlease select an expert to help with the response.\n\nExpert 1: center news media\nExpert 2: left news media\nExpert 3: right news media\nExpert 4: center Reddit\nExpert 5: left Reddit\nExpert 6: right Reddit\n\nExpert:"
            expert_num = lm_utils.llm_response(expert_selection_prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 5)
            for num in range(1, 7):
                if str(num) in expert_num:
                    chosen_expert = num
                    break
            if chosen_expert is None:
                # randomly choose an expert, in the rare event of no choice
                chosen_expert = random.randint(1, 6)
            
            prompt = "Passage: " + comment_pool[str(data[i]["id"])][chosen_expert - 1] + "\n\n" + prompt

            if "steerable_test_valuekaleidoscope" in input_file:
                response = lm_utils.llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 20)
            else:
                response = lm_utils.llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200)
            data[i]["output"] = response
        elif type == "probability":
            prompt = data[i]["question"]

            chosen_expert = None
            expert_selection_prompt = data[i]["question"] + "\n\nPlease select an expert to help with the response.\n\nExpert 1: center news media\nExpert 2: left news media\nExpert 3: right news media\nExpert 4: center Reddit\nExpert 5: left Reddit\nExpert 6: right Reddit\n\nExpert:"
            expert_num, _ = lm_utils.llm_response(expert_selection_prompt, model_name, probs = True, temperature = 0.1, max_new_tokens = 20)
            for num in range(1, 7):
                if str(num) in expert_num:
                    chosen_expert = num
                    break
            if chosen_expert is None:
                # randomly choose an expert, in the rare event of no choice
                chosen_expert = random.randint(1, 6)
            
            prompt = "Passage: " + comment_pool[str(data[i]["id"])][chosen_expert - 1] + "\n\n" + prompt

            response, probs = lm_utils.llm_response(prompt, model_name, probs = True, temperature = 0.1, max_new_tokens = 20)
            output_distribution = [0] * len(data[i]["options"])
            for j in range(len(data[i]["options"])):
                for key in probs:
                    if ind_to_option_map[j] == key.strip():
                        output_distribution[j] += probs[key]
                        break
            if sum(output_distribution) == 0:
                # assign uniform, in the rare event of no option found
                output_distribution = [1 / len(data[i]["options"])] * len(data[i]["options"])
                data[i]["pred_distribution"] = output_distribution
            else:
                # normalize with min-max scaling
                output_distribution = [x / sum(output_distribution) for x in output_distribution]
                data[i]["pred_distribution"] = output_distribution

    with open("output/" + input_file + "_" + model_name + "_moe" ".json", 'w') as f:
        json.dump(data, f, indent = 4)