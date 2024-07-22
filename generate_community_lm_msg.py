import os
import json
import torch
import lm_utils
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "input file name")
    parser.add_argument("-t", "--type", help = "type of operation: generate or probability")
    parser.add_argument("-c", "--checkpoint", default = None, help = "checkpoint path")

    args = parser.parse_args()
    input_file = args.input
    type = args.type
    checkpoint_path = args.checkpoint

    data = None
    with open("input/" + input_file + ".json", 'r') as f:
        data = json.load(f)

    ind_to_option_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    opinionqa_attribute_category_map = {"POLPARTY": "political party", "POLIDEOLOGY": "political ideology", "RELIG": "religion",
                                        "RACE": "race", "EDUCATION": "education", "INCOME": "income",
                                        "CREGION": "region in the United States", "SEX": "Sex"}
    
    if checkpoint_path is not None:
        checkpoint_paths = [checkpoint_path]
    else:
        checkpoint_paths = ["bunsenfeng/mistral-news_l", "bunsenfeng/mistral-news_c", "bunsenfeng/mistral-news_r",
                            "bunsenfeng/mistral-reddit_l", "bunsenfeng/mistral-reddit_c", "bunsenfeng/mistral-reddit_r"]
        # alterntiavely, culture-based community lms
        # checkpoint_paths = ["bunsenfeng/mistral-asia_culture", "bunsenfeng/mistral-europe_culture",
        #                     "bunsenfeng/mistral-northamerica_culture", "bunsenfeng/mistral-southamerica_culture",
        #                     "bunsenfeng/mistral-africa_culture"]

    for checkpoint_path in checkpoint_paths:
        
        # skip if the message already exists
        existing_msgs = os.listdir("community_lm_msgs/")
        if input_file + "_" + checkpoint_path.split("/")[1] + ".json" in existing_msgs:
            continue
    
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            device_map={"": "cuda"},
            torch_dtype=torch.float16,
        )

        generated_msgs = {}
        
        for i in tqdm(range(len(data))):

            if type == "generate":
                prompt = data[i]["input"]

                if "overton" in input_file:
                    prompt = prompt
                if "steerable_test_valuekaleidoscope" in input_file:
                    prompt = "Please respond to the following instruction with explanation. " + prompt

                # msg = text_generator(prompt)

                messages = [
                    {"role": "user", "content": prompt},
                ]

                model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = lora_model.generate(model_inputs, max_new_tokens=200, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=1.0, top_p = 0.9)
                decoded = tokenizer.batch_decode(generated_ids)
                msg = decoded[0].split("[/INST]")[1].replace("</s>", "").strip()

                generated_msgs[data[i]["id"]] = msg
            elif type == "probability":
                prompt = data[i]["question"]

                if "steerable_test_opinionqa" in input_file:
                    attributes = data[i]["attribute"].split("_")
                    prompt = "In terms of " + opinionqa_attribute_category_map[attributes[0]] + ", you are " + attributes[1] + ". Please respond to the following question with explanation. " + prompt
                if "distributional_test_globalopinionqa" in input_file:
                    prompt = "You are from the country of " + data[i]["attribute"] + ", respond to the following instruction with explanation. " + prompt
                if "distributional_test_moralchoice" in input_file:
                    prompt = "Please respond to the following question with explanation. " + prompt

                messages = [
                    {"role": "user", "content": prompt},
                ]

                model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = lora_model.generate(model_inputs, max_new_tokens=200, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=1.0, top_p = 0.9)
                decoded = tokenizer.batch_decode(generated_ids)
                msg = decoded[0].split("[/INST]")[1].replace("</s>", "").strip()

                generated_msgs[data[i]["id"]] = msg

        final_comments_name_part_checkpoint = {
            "bunsenfeng/mistral-news_l": "mistral-news_left",
            "bunsenfeng/mistral-news_c": "mistral-news_center",
            "bunsenfeng/mistral-news_r": "mistral-news_right",
            "bunsenfeng/mistral-reddit_l": "mistral-reddit_left",
            "bunsenfeng/mistral-reddit_c": "mistral-reddit_center",
            "bunsenfeng/mistral-reddit_r": "mistral-reddit_right",
            "bunsenfeng/mistral-asia_culture": "mistral-asia_culture",
            "bunsenfeng/mistral-europe_culture": "mistral-europe_culture",
            "bunsenfeng/mistral-northamerica_culture": "mistral-northamerica_culture",
            "bunsenfeng/mistral-southamerica_culture": "mistral-southamerica_culture",
            "bunsenfeng/mistral-africa_culture": "mistral-africa_culture"
        }

        with open("community_lm_msgs/" + input_file + "_" + final_comments_name_part_checkpoint[checkpoint_path] + ".json", 'w') as f:
            json.dump(generated_msgs, f, indent = 4)