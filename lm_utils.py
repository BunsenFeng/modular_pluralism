from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from tqdm import tqdm
import transformers
import torch
import openai
import os
import time
import numpy as np
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def llm_init(model_name, probs=False):
    global device
    global model
    global tokenizer

    if model_name == "llama2_7b_unaligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model.to(device)

    if model_name == "llama2_7b_aligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model.to(device)

    if model_name == "llama2_13b_unaligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

    if model_name == "llama2_13b_aligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

    if model_name == "llama2_70b_unaligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

    if model_name == "llama2_70b_aligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

    if model_name == "llama3_8b_unaligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_name == "llama3_8b_aligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_name == "llama3_70b_unaligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_name == "llama3_70b_aligned":
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", device_map = "auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_name == "gemma_7b_unaligned":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        model.to(device)

    if model_name == "gemma_7b_aligned":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
        model.to(device)

    if model_name == "chatgpt_unaligned":
        openai.api_key = os.getenv("OPENAI_API_KEY")

    if model_name == "chatgpt_aligned":
        openai.api_key = os.getenv("OPENAI_API_KEY")

def wipe_model():
    global device
    global model
    global tokenizer
    device = None
    model = None
    tokenizer = None
    del device
    del model
    del tokenizer

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200):
    if not model_name == "chatgpt_aligned" and not model_name == "chatgpt_unaligned":
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, do_sample=True, temperature=temperature)

        # print(outputs)

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        generated_text = tokenizer.batch_decode(generated_tokens)[0].strip()

        # next token probability distribution here
        token_probs = {}
        if probs:
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = logits.softmax(dim=-1)
            probabilities = probabilities[0, -1, :]
            top5 = probabilities.topk(10)

            for token, prob in zip(top5.indices, top5.values):
                token_probs[tokenizer.decode(token)] = prob.item()

        if probs:
            return generated_text, token_probs
        else:
            return generated_text

    if model_name == "chatgpt_aligned":
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=10,
        )
        time.sleep(0.1)
        token_probs = {}
        try:
            for key in response["choices"][0]["logprobs"]["top_logprobs"][0].keys():
                tok = key
                score = response["choices"][0]["logprobs"]["top_logprobs"][0][key]
                token_probs[tok] = np.exp(score)
        except:
            # dummy probs
            token_probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}
        if probs:
            return response.choices[0].text.strip(), token_probs
        else:
            return response.choices[0].text.strip()
        
    if model_name == "chatgpt_unaligned":
        response = openai.Completion.create(
                model="davinci-002",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_new_tokens,
                logprobs=10,
            )
        time.sleep(0.1)
        token_probs = {}
        try:
            temp = response["choices"][0]["logprobs"]["top_logprobs"][0]
        except:
            # dummy output if failed, safety filter mostly
            if probs:
                return "N/A", {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
            else:
                return "N/A"
        for key in response["choices"][0]["logprobs"]["top_logprobs"][0].keys():
            tok = key
            score = response["choices"][0]["logprobs"]["top_logprobs"][0][key]
            token_probs[tok] = np.exp(score)
        if probs:
            return response.choices[0].text.strip(), token_probs
        else:
            return response.choices[0].text.strip()
        
def answer_parsing(response):
    # mode 1: answer directly after
    temp = response.strip().split(" ")
    for option in ["A", "B", "C", "D", "E"]:
        if option in temp[0]:
            return option
    # mode 2: "The answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the answer is " + option in temp:
            return option.upper()
    # mode 3: "Answer: A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "answer: " + option in temp:
            return option.upper()
    # mode 4: " A/B/C/D/E " or " A/B/C/D/E."
    for option in ["A", "B", "C", "D", "E"]:
        if " " + option + " " in response or " " + option + "." in response:
            return option
    # mode 5: "The correct answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the correct answer is " + option in temp:
            return option.upper()
    # mode 6: "A: " or "B: " or "C: " or "D: " or "E: "
    for option in ["A", "B", "C", "D", "E"]:
        if option + ": " in response:
            return option
    # mode 7: "A/B/C/D/E" and EOS
    try:
        for option in ["A", "B", "C", "D", "E"]:
            if option + "\n" in response or response[-1] == option:
                return option
    except:
        pass
    # mode 8: (A) or (B) or (C) or (D) or (E)
    for option in ["A", "B", "C", "D", "E"]:
        if "(" + option + ")" in response:
            return option
    # # fail to parse
    # print("fail to parse answer", response, "------------------")
    return "Z" # so that its absolutely wrong