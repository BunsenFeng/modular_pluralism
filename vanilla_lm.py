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

    args = parser.parse_args()
    model_name = args.model
    input_file = args.input
    type = args.type
    portion = args.portion

    if type == "generate":
        lm_utils.llm_init(model_name, probs = False)
    elif type == "probability":
        lm_utils.llm_init(model_name, probs = True)

    data = None
    with open("input/" + input_file + ".json", 'r') as f:
        data = json.load(f)

    ind_to_option_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    if portion < 1:
        data = data[:int(len(data) * portion)]
    
    for i in tqdm(range(len(data))):
        if type == "generate":
            prompt = data[i]["input"]
            if "steerable_test_valuekaleidoscope" in input_file:
                response = lm_utils.llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 20)
            else:
                response = lm_utils.llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200)
            data[i]["output"] = response
        elif type == "probability":
            prompt = data[i]["question"]
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
                output_distribution = [x / sum(output_distribution) for x in output_distribution]
                data[i]["pred_distribution"] = output_distribution

    with open("output/" + input_file + "_" + model_name + "_vanilla" ".json", 'w') as f:
        json.dump(data, f, indent = 4)