import json
import scipy
import argparse
import transformers
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from lm_utils import answer_parsing
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o", "--output", help="output file name")
    argParser.add_argument("-a", "--attribute", default = None, help="attribute to evaluate only")

    args = argParser.parse_args()
    output_file = args.output
    attribute = args.attribute

    data = json.load(open("output/" + output_file + ".json"))

    distance_scores = []
    most_likely_correctness = []
    for item in data:
        if attribute and not attribute in item["attribute"]: # as long as the specified attribute string is in there, we calculate; "EDUCATION" covers all education-related attributes
            continue
        distance_scores.append(scipy.spatial.distance.jensenshannon(item["gold_distribution"], item["pred_distribution"]))
        assert distance_scores[-1] >= 0 and distance_scores[-1] <= 1

        most_likely_correctness.append(item["gold_distribution"].index(max(item["gold_distribution"])) == item["pred_distribution"].index(max(item["pred_distribution"])))

    # print the average and std of scores
    average = sum(distance_scores) / len(distance_scores)
    print("Output file: ", output_file)
    print("Attribute: ", attribute)
    print("Average distance: ", average)
    print("Most likely correctness: ", sum(most_likely_correctness) / len(most_likely_correctness))
    print("-----------------")