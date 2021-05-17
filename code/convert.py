import os
import sys 
import pickle
import json
import argparse
from tqdm import tqdm 
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str)
    parser.add_argument('--gold', type=str)
    parser.add_argument('--type', type=str, choices=['dev', 'test'])
    args = parser.parse_args()
    return args


def convert_format(data):
    new_data = []
    for i, item in tqdm(enumerate(data)):
        one_dict = {}
        for j in range(len(item)):
            if item[j] not in one_dict:
                one_dict[item[j]] = []
            one_dict[item[j]].append(str(j))
        for key in one_dict:
            new_data.append("{}:{}".format(i, " ".join(one_dict[key])))
    return new_data


def main():
    args = get_args()
    with open(args.pred, 'rb') as fin:
        pred = pickle.load(fin)
    with open(args.gold, 'rb') as fin:
        truth = pickle.load(fin)
    time_stamp = re.findall(".+/result_(.+?)/", args.pred)[0]
    print(time_stamp)
    step = re.findall(".+step_(.+?)\.pkl", args.pred)[0]
    print(step)
    output_folder = "convert_res/{}/{}".format(time_stamp, step)
    os.makedirs(output_folder, exist_ok=True)

    converted_pred = convert_format(pred)
    converted_truth = convert_format(truth)

    pred_file = os.path.join(output_folder, "{}_pred.txt".format(args.type))
    with open(pred_file, 'w') as fout:
        for line in converted_pred:
            fout.write(line+"\n")
    truth_file = os.path.join(output_folder, "{}_truth.txt".format(args.type))
    with open(truth_file, 'w') as fout:
        for line in converted_truth:
            fout.write(line+"\n")
    


if __name__ == "__main__":
    main()