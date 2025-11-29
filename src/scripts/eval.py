from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from tqdm import tqdm
import os
import torch
import json

from openai import OpenAI

import asyncio

import argparse

def main():
    parser = argparse.ArgumentParser(description="Program Description")
    # Add command-line arguments
    parser.add_argument('--model_path', type=str, help='Input file path')
    parser.add_argument('--name', type=str, help='Detailed output mode')
    parser.add_argument('--eval_type', type=str, help='Evaluation type')
    parser.add_argument('--eval_batch_size', type=int, help='Evaluation batch size', default=64)
    parser.add_argument('--eval_output_length', type=int, help='Evaluation output length', default=256)
    parser.add_argument('--eval_input_truncate_length', type=int, help='Evaluation input truncate length', default=256)
    parser.add_argument('--base_url', type=str, help='Base URL for API', default='your_base_url')
    parser.add_argument('--api_key', type=str, help='API key', default='your_api_key')
    args = parser.parse_args()
    # Write a unified system prompt

    with open(f"data_eval/{args.name}.json", 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    model = AutoModelForCausalLM.from_pretrained(
                args.model_path, device_map = "auto",trust_remote_code=True, dtype="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, trust_remote_code=True, padding_side="left"
    )
    adapter_path = os.path.join(args.model_path,"adp")
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(
            model, adapter_path
    )
    model.eval()

    results = get_result(model, tokenizer, eval_data,
        batch_size=args.eval_batch_size,
        output_length=args.eval_output_length,
        input_truncate_length=args.eval_input_truncate_length
    )

    if args.eval_type != "None":
        if args.eval_type == "classification":
            pass
        elif args.eval_type == "generation":
            pass
    
    with open(f"data_result/{args.name}_{args.eval_type}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def get_result(model, tokenizer, dataset, 
    batch_size=64, output_length=256, input_truncate_length=256):
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches", total=len(dataset)//batch_size + (1 if len(dataset) % batch_size != 0 else 0)):
            batch_data = dataset[i:i + batch_size]
            batch_inputs = [item["input"] for item in batch_data]
            
            # Encoding inputs
            inputs = tokenizer(
                batch_inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=input_truncate_length,
                pad_token_id=tokenizer.eos_token_id
            )
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generating outputs
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=output_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            new_outputs = []
            for out_text, item in zip(batch_outputs, batch_data):
                # Check if the generated output starts with the input and remove it
                new_out = out_text[len(item["input"]):].strip()  # Remove the input part from the start
                new_outputs.append(
                    {
                        "input": item["input"],
                        "new_out": new_out,
                        "label": item["label"],
                    }
                )
            results.extend(new_outputs)
    return results

if __name__ == "__main__":
    main()
