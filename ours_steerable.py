import json
import random
import lm_utils
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help = "model name")
    parser.add_argument("-i", "--input", help = "input file name")
    parser.add_argument("-t", "--type", help = "generate or probability or generate-special") # generate for Value Kaleidoscope-like task, probability for OpinionQA-like task
    parser.add_argument("-o", "--portion", default = 1, help = "portion of the data to use")
    parser.add_argument("-c", "--community_setting", default = "perspective", help = "community setting") # perspective or culture or mixed

    args = parser.parse_args()
    model_name = args.model
    input_file = args.input
    type = args.type
    portion = float(args.portion)
    community_setting = args.community_setting

    if type == "generate" or type == "generate-special":
        lm_utils.llm_init(model_name, probs = False)
    elif type == "probability":
        lm_utils.llm_init(model_name, probs = True)

    temp_input_file = input_file
    if "_small" in input_file:
        temp_input_file = input_file.replace("_small", "")

    comment_pool = {}
    domain_list = []
    if community_setting == "perspective":
        domain_list = ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                       "mistral-reddit_center", "mistral-reddit_left", "mistral-reddit_right"]
    elif community_setting == "culture":
        domain_list = ["mistral-africa_culture", "mistral-asia_culture", "mistral-europe_culture", "mistral-northamerica_culture", "mistral-southamerica_culture"]
    elif community_setting == "mixed":
        domain_list = ["mistral-news_center", "mistral-news_left", "mistral-news_right",
                       "mistral-africa_culture", "mistral-asia_culture", "mistral-southamerica_culture"]
    for domain in domain_list:
        comments = json.load(open("community_lm_msgs/" + temp_input_file + "_" + domain + ".json", 'r'))
        for key in comments.keys():
            if key not in comment_pool:
                comment_pool[key] = [comments[key]]
            else:
                comment_pool[key].append(comments[key])

    ind_to_option_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    opinionqa_attribute_category_map = {"POLPARTY": "political party", "POLIDEOLOGY": "political ideology", "RELIG": "religion",
                                        "RACE": "race", "EDUCATION": "education", "INCOME": "income",
                                        "CREGION": "region in the United States", "SEX": "Sex"}

    data = None
    with open("input/" + input_file + ".json", 'r') as f:
        data = json.load(f)

        if portion < 1:
            data = data[:int(len(data) * portion)]

        for item in tqdm(data):
            item["comments"] = comment_pool[str(item["id"])]
            if "opinionqa" in input_file:
                attributes = item["attribute"].split("_")
                prompt1 = "Which of the following comments best reflect the people of " + attributes[1] + " in terms of " + opinionqa_attribute_category_map[attributes[0]] + "?\n\n"
                for i in range(len(item["comments"])):
                    prompt1 += "Comment " + str(i + 1) + ": " + item["comments"][i] + "\n\n"
                prompt1 += "Please select one comment number from 1 to " + str(len(item["comments"])) + ":"
            elif "valuekaleidoscope" in input_file:
                attribute = item["vrd"]
                prompt1 = "Which of the following comments best reflect the value of " + attribute + "?\n\n"
                for i in range(len(item["comments"])):
                    prompt1 += "Comment " + str(i + 1) + ": " + item["comments"][i] + "\n\n"
                prompt1 += "Please select one number from 1 to " + str(len(item["comments"])) + ":"

            if type == "generate":
                response1 = lm_utils.llm_response(prompt1, model_name, probs = False, temperature = 0.1, max_new_tokens = 20)
            elif type == "probability":
                response1, _ = lm_utils.llm_response(prompt1, model_name, probs = True, temperature = 0.1, max_new_tokens = 20)

            the_comment = None
            for i in range(len(item["comments"])):
                if str(i + 1) in response1:
                    the_comment = item["comments"][i]
                    break
            if the_comment is None:
                # randomly chosse one in the rare event of no choice
                the_comment = random.choice(item["comments"])

            if "opinionqa" in input_file:
                attributes = item["attribute"].split("_")
                prompt2 = "In terms of " + opinionqa_attribute_category_map[attributes[0]] + ", you are " + attributes[1] + ". Please respond to the following question with the help of a passage.\n\nPassage: " + the_comment + "\n\n" + item["question"]
            elif "valuekaleidoscope" in input_file:
                attribute = item["vrd"]
                prompt2 = "Answer the question with the help of a passage. Passage: " + the_comment + "\n\n" + item["input"]

            if type == "generate":
                response2 = lm_utils.llm_response(prompt2, model_name, probs = False, temperature = 0.1, max_new_tokens = 20)
                item["output"] = response2
            elif type == "probability":
                response2, probs = lm_utils.llm_response(prompt2, model_name, probs = True, temperature = 0.1, max_new_tokens = 20)
                output_distribution = [0] * len(item["options"])
                for j in range(len(item["options"])):
                    for key in probs:
                        if ind_to_option_map[j] == key.strip():
                            output_distribution[j] += probs[key]
                            break
                if sum(output_distribution) == 0:
                    # assign uniform in the rare event of no option found
                    output_distribution = [1 / len(item["options"])] * len(item["options"])
                    item["pred_distribution"] = output_distribution
                else:
                    output_distribution = [x / sum(output_distribution) for x in output_distribution]
                    item["pred_distribution"] = output_distribution

    if community_setting == "perspective":
        with open("output/" + input_file + "_" + model_name + "_ours.json", 'w') as f:
            json.dump(data, f, indent = 4)
    else:
        with open("output/" + input_file + "_" + model_name + "_ours_" + community_setting + ".json", 'w') as f:
            json.dump(data, f, indent = 4)