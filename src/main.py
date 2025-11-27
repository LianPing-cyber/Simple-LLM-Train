from data_manager import Data_Normal
from model import Model
from train import Trainer
from scripters import Evaluater

import torch
import argparse

import asyncio

def gpu_clear():
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Program Description")
    # Add command-line arguments
    parser.add_argument('--data', type=str, help='Input file path')
    parser.add_argument('--task_name', type=str, help='Detailed output mode')
    parser.add_argument('--do_train', type=int, help="0 means no training, only evaluation; 1 means training + evaluation", default=0)
    parser.add_argument('--continue_train', type=int, help="0 means new start; 1 means continue training", default=1)

    parser.add_argument('--original_model_folder', type=str, help='Folder of the original model')
    parser.add_argument('--output_model_folder', type=str, help='Folder for saving the model')
    parser.add_argument('--epoch_num', type=int, help='Number of training epochs', default=1)
    parser.add_argument('--cutoff_len', type=int, help='Cutoff length for input sequences', default=1024)
    parser.add_argument('--train_stage', type=str, help='Training stage', default='sft')
    parser.add_argument('--finetuning_type', type=str, help='Type of fine-tuning: full or lora', default='lora')

    parser.add_argument('--train_num', type=int, help='How much data you want to use for training', default=0)
    parser.add_argument('--eval_num', type=int, help='How much data you want to use for evaluation', default=0)
    parser.add_argument('--eval_batch_size', type=int, help='Batch size for evaluation', default=64)
    parser.add_argument('--train_template', type=str, help='Template name for training data', default='empty')
    parser.add_argument('--eval_template', type=str, help='Template name for evaluation data', default='empty')
    parser.add_argument('--eval_output_length', type=int, help='Evaluation output length', default=256)

    parser.add_argument('--lf_path',type=str, help='Llama_factory folder path, e.g., /root/llama_factory/')
    parser.add_argument('--lf_data_dir', type=str, help='Where to save llama_factory datasets')
    parser.add_argument('--base_url', type=str, help='Base URL for API', default='your_base_url')
    parser.add_argument('--api_key', type=str, help='API key', default='your_api_key')


    parser.add_argument('--per_device_train_batch_size', type=int, help='Batch size for each GPU', default=1)
    parser.add_argument('--nproc_per_node', type=int, help='Parallelization setting for number of processes per node during training', default=2)
    parser.add_argument('--finetuning_type', type=str, help='finetuning type: full or lora', default='lora')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=5e-5)

    args = parser.parse_args()

    data = args.data
    task_name = args.task_name
    do_train = args.do_train
    
    epoch_num = args.epoch_num
    train_num = args.train_num
    eval_num = args.eval_num
    eval_batch_size = args.eval_batch_size

    original_model = args.original_model_folder
    output_model = args.output_model_folder

    lf_path = args.lf_path
    lf_data_dir = args.lf_data_dir

    data = Data_Normal(task_name, data, train_num, eval_num)
    data.template_train(args.train_template)
    data.template_eval(args.eval_template)

    if do_train == 1:
        data.train_write()
        trainer = Trainer(lf_path, lf_data_dir)
        trainer.get_parameters(stage=args.train_stage, 
                finetuning_type=args.finetuning_type, 
                max_sample=train_num,
                per_device_train_batch_size = args.per_device_train_batch_size,
                nproc_per_node=args.nproc_per_node,
                learning_rate=args.learning_rate,
                cutoff_len=args.cutoff_len
                )
        model = Model(original_model)
        if args.continue_train == 1:
            model.continue_train_task(task_name, trainer)
        else:
            model.new_train_task(output_model, task_name, trainer)
        for epoch in range(epoch_num):
            model.train(1)

    data.eval_write()
    evaluater = Evaluater(output_model, task_name)
    evaluater.get_parameters(
        eval_batch_size=eval_batch_size,
        eval_input_truncate_length=args.cutoff_len,
        eval_output_length=args.eval_output_length,
        base_url = args.base_url,
        api_key = args.api_key)

if __name__ == "__main__":
    main()
