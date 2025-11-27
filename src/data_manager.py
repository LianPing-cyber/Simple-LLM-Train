import json
import os
from templates import get_template

import random

class Data_Nomral():
    def __init__(self, name, data, train_num, eval_num):
        self.name = name
        self.data = data
        self.train_num = train_num
        self.eval_num = eval_num
        with open(data, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        random.shuffle(dataset)
        self.dataset = dataset
        if train_num + eval_num > len(dataset):
            print(f"Requested split exceeds total data length ({len(dataset)}).")
        self.train_data = dataset[:train_num]
        self.eval_data = dataset[train_num:train_num + eval_num]
        self.templated_train_data = []
        self.templated_eval_data = []

    def reset_train_data(self):
        self.train_data = self.dataset[:self.train_num]
        
    def template_train(self, template_name):
        self.templated_train_data = []
        template = get_template(template_name)
        for item in self.train_data:
            new_input = template.render(input = item['input'])
            new_output = item["label"]
            self.templated_train_data.append({
                "input": new_input,
                "label": new_output
            })
    
    def template_eval(self, template_name):
        self.templated_eval_data = []
        template = get_template(template_name)
        for item in self.eval_data:
            new_input = template.render(input = item['input'])
            new_output = item["label"]
            self.templated_eval_data.append({
                "input": new_input,
                "label": new_output
            })
    
    def train_write(self):
        train_alpaca = []
        for item in self.templated_train_data:
            alpaca_item = {
                "instruction": item["input"],
                "input": "",
                "output": item["output"]
            }
            train_alpaca.append(alpaca_item)
        file_name = f"train_data/{self.name}.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(train_alpaca, f, ensure_ascii=False, indent=2)
        add_info(self.name, file_name, "alpaca")

    def eval_write(self):
        file_name = f"eval_data/{self.name}_eval.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.templated_eval_data, f, ensure_ascii=False, indent=2)
     
class Data_DPO():
    def __init__(self, name, data, train_num, eval_num):
        self.name = name
        self.data = data
        self.train_num = train_num
        self.eval_num = eval_num
        with open(data, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        random.shuffle(dataset)
        self.dataset = dataset
        if train_num + eval_num > len(dataset):
            print(f"Requested split exceeds total data length ({len(dataset)}).")
        self.train_data = dataset[:train_num]
        self.eval_data = dataset[train_num:train_num + eval_num]
        self.templated_train_data = []
        self.templated_eval_data = []

    def reset_train_data(self):
        self.train_data = self.dataset[:self.train_num]
        
    def template_train(self, template_name):
        self.templated_train_data = []
        template = get_template(template_name)
        for item in self.train_data:
            new_input = template.render(input = item['input'])
            chosen_output = item["chosen"]
            rejected_output = item["rejected"]
            self.templated_train_data.append({
                "input": new_input,
                "chosen": chosen_output,
                "rejected": rejected_output
            })

    def template_eval(self, template_name):
        self.templated_eval_data = []
        template = get_template(template_name)
        for item in self.eval_data:
            new_input = template.render(input = item['input'])
            label = item["chosen"]
            self.templated_eval_data.append({
                "input": new_input,
                "label": label,
            })

    def train_write(self):
        train_dpo = []
        for item in self.templated_train_data:
            dpo_item = {
                "instruction": item["input"],
                "input": "",
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            }
            train_dpo.append(dpo_item)
        file_name = f"train_data/{self.name}_dpo.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(train_dpo, f, ensure_ascii=False, indent=2)
        add_info(self.name, file_name, "dpo")

    def eval_write(self):
        file_name = f"eval_data/{self.name}_eval.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.templated_eval_data, f, ensure_ascii=False, indent=2)

def add_info(data_name, data_path, data_type):
    """Register dataset information in the dataset_info.json file"""
    dir_path = os.path.dirname(data_path)
    file_name = os.path.basename(data_path)
    info_path = os.path.join(dir_path, "dataset_info.json")
        
    # Read existing config or create new config
    try:
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
                dataset_info = {}
    except Exception as e:
        print(f"Failed to read dataset_info.json: {e}")
        dataset_info = {}
        
    # Update Alpaca format dataset information
    if data_type == "alpaca":
        dataset_info[data_name] = {
            "file_name": file_name,
            "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                }
        }
    try:
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"Dataset information has been updated: {info_path}")
        return True
    except Exception as e:
        print(f"Failed to update dataset_info.json: {e}")
        return False