import argparse
import csv
import numpy as np
import os
import torch
import random
import sys


csv.field_size_limit(sys.maxsize)


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_csv_questions(filename):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets', filename)
    questions = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            question, answer = row
            questions.append([question, answer])
    return questions


def log_to_file(conversation, curr_datetime, info_run):
    filepath = os.path.join(os.path.dirname(__file__), "..", "tests", "outputs", f"output_{curr_datetime}.txt")
    with open(filepath, 'a', encoding='utf-8') as file:
        file.write('INFORMATION ON THE RUN\n\n')
        for key in info_run.keys():
            file.write(f"{key}: {info_run[key]}\n")
        file.write('\n-----------------------------------\n\n')
        file.write(conversation)
