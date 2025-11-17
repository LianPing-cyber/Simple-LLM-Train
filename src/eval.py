from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from tqdm import tqdm
import os
import torch
import re

class Evaluate_6_Class():
    def __init__(self, model_path, dataset, dtype="bfloat16"):
        self.model_path=model_path
        self.dataset=dataset.eval_dataset

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map = "auto",trust_remote_code=True, dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left"
        )

        adapter_path = os.path.join(self.model_path,"/adp")
        if os.path.exists(adapter_path):
            self.model = PeftModel.from_pretrained(
                self.model_path, adapter_path
        )
        self.model.eval()
        

    def generate_result(self, number=-1, batch_size=64):
        if number == -1:
            data = self.dataset
        elif number >= len(self.dataset):
            print(f"要求测试的数据数量--{number}超出加载dataset的总长度--{len(self.dataset)}，使用全部测试")
            data = self.dataset
        else:
            data = self.dataset[:number]
        
        #data是一个大list，每个元素为一个字典，含“input”和“output”两个键值
        #请为每个input生成generated_output内容，并添加到相应的字典内
        all_outputs = []

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", total=len(data)//batch_size + (1 if len(data) % batch_size != 0 else 0)):
            batch_data = data[i:i + batch_size]
            batch_inputs = [item["input"] for item in batch_data]
            
            # 编码输入
            inputs = self.tokenizer(
                batch_inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=256
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_outputs = [out_text.replace(in_text, "") for out_text, in_text in zip(batch_outputs, batch_inputs)]
            all_outputs.extend(batch_outputs)      
        for out_text, item in zip(all_outputs, data):
            '''print("***"*10)
            print("input:", item["input"],"\n")
            print("generated:", out_text)
            print("label:", item["output"])'''
            item["generated_output"]=out_text
        self.result = data

    def eval_6class(self):
        # 统计 6 个类别的总数和正确数（0~5）
        safety_total = [0.0000001] * 3
        safety_right = [0.0] * 4

        category_total = [0.0000001] * 5
        category_correct = [0.0] * 5

        # 预编译正则：匹配 [SAFE]-0 或 [UNSAFE]-1~5
        pattern = re.compile(r'(安全|危险)\s*-\s*([0-5])')

        def match_result(text: str):
            """
            从文本中解析出 (safety, num)
            期望格式: [SAFE]-0 或 [UNSAFE]-x
            解析失败则返回 (None, None)
            """
            if text is None:
                return None, None
            m = pattern.search(text)
            if not m:
                return None, None
            safety, num = m.group(1), m.group(2)
            return safety, num

        for item in self.result:
            label_text = item.get("output", "")
            gen_text = item.get("generated_output", "")

            # 解析标签和生成结果
            label_safety, label_num = match_result(label_text)
            gen_safety, gen_num = match_result(gen_text)

            if label_safety is None or label_num is None:
                print("无效空label")
                safety_total[2]+=1
                continue
            elif label_safety == "安全" and label_num != "0":
                print("safety错标label")
                safety_total[2]+=1
                continue
            
            if label_safety == "安全":
                safety_total[0]+=1
                if gen_safety is None or gen_num is None:
                    safety_right[2]+=1
                    continue
                elif gen_safety == "安全" and gen_num != "0":
                    safety_right[3]+=1
                    continue
                elif gen_safety == "安全":
                    safety_right[0]+=1
            
            elif label_safety == "危险":
                safety_total[1]+=1
                category_total[int(label_num)-1]+=1
                if gen_safety is None or gen_num is None:
                    safety_right[2]+=1
                    continue
                elif gen_safety == "安全" and gen_num != "0":
                    safety_right[3]+=1
                    continue
                if gen_safety == "危险":
                    safety_right[1]+=1
                    if label_num == gen_num:
                        category_correct[int(gen_num)-1]+=1

        accuracy_safety = (safety_right[0]+safety_right[1])/(safety_total[0]+safety_total[1])
    
        accuracy_all = [safety_right[0]/safety_total[0], category_correct[0]/category_total[0],
                category_correct[1]/category_total[1], category_correct[2]/category_total[2],
                category_correct[3]/category_total[3], category_correct[4]/category_total[4]]
        accuracy_num = [safety_total[0], category_total[0], category_total[1], category_total[2],
                        category_total[3], category_total[4]]
        
        accuracy_category_average = sum(accuracy_all)/6
        accuracy_category_weight = sum([acc * num / (safety_total[0] + safety_total[1]) for acc, num in zip(accuracy_all, accuracy_num)])
        
        print(f"""安全性准确率为：{accuracy_safety}
类均准确率为：{accuracy_category_average}
类加权准确率（总准确率）为：{accuracy_category_weight}""")
        
        print(f"""
无效label数量：**{safety_total[2]}**
======================================
类别0：正确数-{safety_right[0]}；总数-{int(safety_total[0])}；正确率-{accuracy_all[0]}
类别1：正确数-{category_correct[0]}；总数-{int(category_total[0])}；正确率-{accuracy_all[1]}
类别2：正确数-{category_correct[1]}；总数-{int(category_total[1])}；正确率-{accuracy_all[2]}
类别3：正确数-{category_correct[2]}；总数-{int(category_total[2])}；正确率-{accuracy_all[3]}
类别4：正确数-{category_correct[3]}；总数-{int(category_total[3])}；正确率-{accuracy_all[4]}
类别5：正确数-{category_correct[4]}；总数-{int(category_total[4])}；正确率-{accuracy_all[5]}
======================================
dizzy time: {safety_right[2]}
silence time: {safety_right[3]}
""")

        # 返回统计结果供后续使用
        return {
            'overall_accuracy': accuracy_category_weight,
            'accuracy_safety': accuracy_safety,
            'dizzy_time': safety_right[2],
            'silence_time': safety_right[3]
        }
