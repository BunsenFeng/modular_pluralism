import json
import random
import argparse
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from lm_utils import answer_parsing
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o", "--output", help="output file name")

    args = argParser.parse_args()
    output_file = args.output

    data = json.load(open("output/" + output_file + ".json"))
    pred = []
    gold = []
    option_to_number_map = {"A": 0, "B": 1, "C": 2}
    for item in tqdm(data):
        gold.append(option_to_number_map[item["label"]])
        now_pred = answer_parsing(item["output"])
        if now_pred in option_to_number_map:
            pred.append(option_to_number_map[now_pred])
        else:
            if "support" in item["output"].lower():
                pred.append(0)
            elif "oppose" in item["output"].lower():
                pred.append(1)
            elif "either" in item["output"].lower() or "it depends" in item["output"].lower():
                pred.append(2)
            else:
                # put a random guess, in the rare cases
                pred.append(3)
    
    print("Accuracy:", accuracy_score(gold, pred))
    print("Balanced accuracy:", balanced_accuracy_score(gold, pred))
    print("Macro F1 score:", f1_score(gold, pred, average="macro"))
    print("Micro F1 score:", f1_score(gold, pred, average="micro"))

    # binary setting where "either" options and predictions are removed
    pred_binary = []
    gold_binary = []
    for i in range(len(pred)):
        if pred[i] == 2 or gold[i] == 2:
            continue
        pred_binary.append(pred[i])
        gold_binary.append(gold[i])
    
    print("Binary accuracy:", accuracy_score(gold_binary, pred_binary))
    print("Binary balanced accuracy:", balanced_accuracy_score(gold_binary, pred_binary))
    print("Binary macro F1 score:", f1_score(gold_binary, pred_binary, average="macro"))
    print("Binary micro F1 score:", f1_score(gold_binary, pred_binary, average="micro"))