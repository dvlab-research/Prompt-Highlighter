# this code is for evaluation of the mmbench stage 1.

import json
import re
from collections import defaultdict
from icecream import ic

# define a function to read the jsonl file
def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data

# use arg to input data_name
import sys
# data_name = sys.argv[1]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="base_models/LLaVA/playground/data/eval/mmbench/answers/mmbench_dev_20230712/llava-v1.5-13b-hl-1.3-2.0-0.01.jsonl")
args = parser.parse_args()
data_name = args.filename
print(data_name)

data = [line for line in (read_jsonl(f'{data_name}'))]

questions = defaultdict(list)

# iterate over the data and group the questions
for d in data:
    question_id = d['question_id'] % 1000000
    questions[question_id].append(d)

# initialize the counter for the correct questions
correct_questions = 0
single_match_num = 0
single_correct_num = 0
categories = {}
# iterate over the groups of questions
for _, group in questions.items():
    correct = True
    for qid, question in enumerate(group):
        matched_answer = []
        answer = question['answer']
        # verify whether it is a list:
        if isinstance(question['text'], list):
            out = question['text'][0]
        else:
            out = question['text']
        potential_answers = []
        matched = False
        pattern = re.compile('\\b[A-Z]\\b')
        matched_answer = re.findall(pattern, out)

        if len(matched_answer) > 0:
            # if len(set(matched_answer)) == 1:
            single_match_num += 1
            if matched_answer[0] != question['answer']:
                correct = False
        else:
            correct = False
            
        if qid == 0 and correct:
            single_correct_num += 1
        
    if correct:
         correct_questions += 1

# print the overall performance
print(f'The model answered correctly {single_correct_num} out of {len(questions)} questions in single round.')
print(f'The model answered correctly {correct_questions} out of {len(questions)} questions in circular round.')
print(f'The model matched correctly {single_match_num} out of {len(data)} questions.')
print(correct_questions/len(questions))