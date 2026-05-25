import json
import torch
import random
import os
import numpy as np
import re

def load_data(path):
    try:
        return json.load(open(path, 'r'))
    except:
        with open(path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    return data


def save_data(data, path, jsonl=False):
    if jsonl:
        if type(data) == dict:
            data = [data]
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        json.dump(data, open(path, 'w'), indent=2)

def read_file(path):
    with open(path, 'r') as f:
        txt = f.read()
    return txt

def set_seed(seed):
    """
    :param seed:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_dir(path, creat=True, force=False):
    path = os.path.split(path)[0]
    print(f'Checking {path}')
    if not os.path.exists(path):
        if creat:
            os.makedirs(path)
            print('Folder %s has been created.' % path)
            return True
        else:
            return False
    else:
        if force:
            os.makedirs(path)
            print('Force to create %s.' % path)
        return True

def check_file(path):
    if os.path.exists(path):
        return True
    return False

def replace_with_terms(prefix:str, content:dict):
    output = prefix
    for term,txt in content.items():
        output = output.replace(term, txt)
    return output

def extract_code_chunk(input_text):
    """
    Extract the first content inside a code block that starts with either ```cpp or ```
    """
    code_block_pattern = re.compile(r"```(?:cpp)?\n(.*?)\n```", re.DOTALL)
    match = code_block_pattern.search(input_text)
    if match:
        return match.group(1)
    return None
