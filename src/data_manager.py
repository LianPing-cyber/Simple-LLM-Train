import json
import os
from templates import template_normal

import random

class Data_Normal():
    """Data management class: handles the loading, transformation, and registration of content classification and conversation data"""
    
    def __init__(self, data_name, data_path, data_type):
        """Initialize the data manager"""
        self.data_name = data_name
        self.data_path = data_path
        self.data_type = data_type
        self.template = template_normal
        self.dataset = []
        self.eval_dataset = []
    
    def get_data(self, max_num=None, random_sample=0):
        """Load and process raw data, supports content and conversation data formats"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
        
        # Handle data length
        total_length = len(data_list)
        if max_num is not None and max_num < total_length:
            if random_sample == 1:
                data_list = random.sample(data_list, max_num)
                print(f"Randomly selected {max_num} data entries")
            else:
                data_list = data_list[:max_num]
                print(f"Selected the first {max_num} data entries")
        elif max_num is not None and max_num > total_length:
            data_list = data_list[:-10000]
            print(f"Data length ({total_length}) is less than the maximum limit ({max_num})")
        data_get = []
        # Process data based on type
        for item in data_list:
            if self.data_type == "content":
                input_text = item.get("input", "")
                label = item.get("output", "")
                
            elif self.data_type == "conversation":
                conversation = item.get("conversation", [])
                input_text = ""
                for turn in conversation:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    input_text += f"{role}:{content}\n"
                label = item.get("output", "")
            else:
                print(f"Unsupported data type: {self.data_type}")
                return False
            
            # Label processing: 0 for safe, others for unsafe
            output = label

            data_get.append({
                "input": self.template(input_text),
                "output": output
            })

        self.dataset = data_get
        print(f"Successfully loaded {len(self.dataset)} data entries")
        return True

    def data_balance(self):
        '''Balance the dataset by output category, find the least frequent class and align the other classes, then shuffle the dataset and update dataset'''
        
        # Step 1: Categorize by output
        classes = {}
        for data in self.dataset:
            output = data['output']
            if output not in classes:
                classes[output] = []
            classes[output].append(data)
        
        # Step 2: Find the smallest class
        min_class_count = min(len(data_list) for data_list in classes.values())
        
        # Step 3: Trim each class's data to match the smallest class count
        balanced_data = []
        for class_data in classes.values():
            balanced_data.extend(random.sample(class_data, min_class_count))  # Randomly select samples
        
        # Step 4: Shuffle the dataset
        random.shuffle(balanced_data)
        
        # Step 5: Update dataset and report new length
        self.dataset = balanced_data
        print(f"Dataset length after balancing: {len(self.dataset)}")
    
    def get_eval_data(self, max_num):
        """Load and process raw data, supports content and conversation data formats"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
        print(len(self.dataset))
        data_list = data_list[len(self.dataset):]

        # Handle data length
        total_length = len(data_list)
        if max_num is not None and max_num < total_length:
            data_list = random.sample(data_list, max_num)
            print(f"Data has been truncated to {max_num} entries")
        elif max_num is not None and max_num >= total_length:
            print(f"Data length ({total_length}) is less than the maximum limit ({max_num}), selecting all data")
            data_list = data_list
        data_get = []
        # Process data based on type
        for item in data_list:
            if self.data_type == "content":
                input_text = item.get("input", "")
                label = item.get("output", "")
                
            elif self.data_type == "conversation":
                conversation = item.get("conversation", [])
                input_text = ""
                for turn in conversation:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    input_text += f"{role}:{content}\n"
                label = item.get("output", "")
            else:
                print(f"Unsupported data type: {self.data_type}")
                return False
            
            # Label processing: 0 for safe, others for unsafe
            output = label
            
            data_get.append({
                "input": self.template(input_text),
                "output": output
            })

        self.eval_dataset = data_get
        print(f"Successfully loaded {len(self.eval_dataset)} data entries")
        return True
    

    def write_in_alpaca(self, save_file):
        """Convert the data to Alpaca format and save it"""
        alpaca_data = []
        for item in self.dataset:
            alpaca_item = {
                "instruction": item["input"],
                "input": "",
                "output": item["output"]
            }
            alpaca_data.append(alpaca_item)
        
        try:
            # Check if the directory exists
            save_dir = os.path.dirname(save_file)
            if save_dir and not os.path.exists(save_dir):
                raise FileNotFoundError(f"Path does not exist: {save_dir}")
            
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
            
            # Register dataset information
            self.add_info(self.data_name, save_file, "alpaca")
            print(f"Data has been saved to: {save_file}")
            return True
        except Exception as e:
            print(f"Failed to save data: {e}")
            return False
    
    def add_info(self, data_name, data_path, data_type):
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
        
        # Save the updated config
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            print(f"Dataset information has been updated: {info_path}")
            return True
        except Exception as e:
            print(f"Failed to update dataset_info.json: {e}")
            return False