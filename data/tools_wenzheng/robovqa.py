import json

path = "/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA/robovqa_reasoning_0.json"


path_1 = "/storage/wenzheng/showrobot/hf_download/datasets/raw/RoboVQA/robovqa_understanding.json"

info = json.load(open(path, 'r'))

info_1 = json.load(open(path_1, 'r'))
print(1)