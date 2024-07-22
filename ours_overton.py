import json
import lm_utils
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help = "model name")
    parser.add_argument("-i", "--input", help = "input file name")
    parser.add_argument("-t", "--type", help = "generate or probability") # generate for Value Kaleidoscope-like task, probability for OpinionQA-like task
    parser.add_argument("-o", "--portion", default = 1, help = "portion of the data to use")
    parser.add_argument("-c", "--community_setting", default = "perspective", help = "community setting") # perspective or culture or mixed

    args = parser.parse_args()
    model_name = args.model
    input_file = args.input
    type = args.type
    portion = float(args.portion)
    community_setting = args.community_setting

    if type == "generate":
        lm_utils.llm_init(model_name, probs = False)
    elif type == "probability":
        print("there shouldn't be probability-based evaluation for overton!")
        exit(0)

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

    data = None
    with open("input/" + input_file + ".json", 'r') as f:
        data = json.load(f)

        if portion < 1:
            data = data[:int(len(data) * portion)]

        for item in tqdm(data):
            item["comments"] = comment_pool[str(item["id"])]
            prompt = "Please comment on a given situation with the help of the following passages. Make sure to reflect diverse values and perspectives.\n\n"
            prompt += "Situation: " + item["situation"] + "\n\n"
            for i in range(len(item["comments"])):
                prompt += "Passage " + str(i + 1) + ": " + item["comments"][i] + "\n\n"
            prompt += "Comment:"
            
            if type == "generate":
                response = lm_utils.llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200)
                item["output"] = response
            if type == "probability":
                print("there shouldn't be probability-based evaluation for overton!")
                exit(0)

    if community_setting == "perspective":
        with open("output/" + input_file + "_" + model_name + "_ours.json", 'w') as f:
            json.dump(data, f, indent = 4)
    else:
        with open("output/" + input_file + "_" + model_name + "_ours_" + community_setting + ".json", 'w') as f:
            json.dump(data, f, indent = 4)