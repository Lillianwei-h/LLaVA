import json
import base64
import os
import re
from prompts import get_system_prompt
import random

DATA_PATH  = "../../data/ManytoMany_all"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def process_asking_content(d, all_dict, mode, i, dataset):
    system_prompt = get_system_prompt(dataset)

    all_dict[i]['id'] = d['id']

    question = d['conversations'][0]['content']
    all_dict[i]['question'] = question
    prompt = ""
    prompt += system_prompt+'\n'
    prompt += "### Question:\n"

    for q in question:
        if q['text'] is not None:
            prompt+=q['text']+'\n'
        if q['image'] is not None:
            image_path = os.path.join(DATA_PATH, dataset, q['image'])
            if os.path.exists(image_path):
                image_file = image_path
            else:
                print(f"File {image_path} doesn't exist.")

    all_dict[i]['gt_answer'] = d['conversations'][1]['content']
    all_dict[i]['model'] = "llava"
    all_dict[i]['prompt'] = prompt
    all_dict[i]['image_file'] = image_file
    

def get_data(filepath, dataset, trucate_len = None, mode = "ans", sample_radio: float = None):
    print(f"Using {mode} mode.")

    with open(filepath, 'r') as f:
        data = json.load(f)

    if trucate_len != 0:
        data = data[:trucate_len]

    if sample_radio is not None:
        indexed_data = list(enumerate(data))
        sampled_data = random.sample(indexed_data, round(sample_radio * len(data)))
        sampled_indices = [index for index, value in sampled_data]
        sampled_values = [value for index, value in sampled_data]
        data = sampled_values
        with open(os.path.join(os.path.dirname(filepath),"temp","gpt_annotation.json"), 'w') as f:
            json.dump(sampled_indices,f)
        print(f"Selected sample num: {len(data)}")
        

    all_dict = {}
    for j in range(len(data)):
        all_dict[j] = {}

    i=0
    for d in data:
        process_asking_content(d, all_dict, mode, i, dataset)
        i+=1

    return all_dict

def decode_answer(answer):
    match = re.search(r'<SCORE:\s*([-+]?\d*\.?\d+)>', answer) # change it to fit your own rule
    if match:
        return float(match.group(1))
    else:
        return None
