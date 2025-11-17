import os
import shutil
import sys

class Model():
    def __init__(self, original_model) -> None:
        self.original_model = original_model

    def new_train_task(self, output_model, task_name, trainer, evaluater=None):
        self.new_copy_for_original_model(output_model)
        self.task_name = task_name
        self.trainer = trainer
        self.evaluater = evaluater

    def train(self, epoch):
        self.trainer.get_source(self)
        self.trainer.train(epoch)

    def new_copy_for_original_model(self, output_model):
        self.output_model = output_model
        # 新建训练任务，创建原模型副本，防止污染
        self.output_model = output_model
        try:
            # 检查a路径是否存在
            if not os.path.exists(self.original_model):
                print(f"错误：源文件夹 '{self.original_model}' 不存在")
                return False
            
            # 检查b路径是否存在
            if os.path.exists(output_model):
                print(f"目标文件夹 '{output_model}' 已存在，正在删除...")
                try:
                    # 删除b路径（包括所有子文件和文件夹）
                    if os.path.isfile(output_model):
                        os.remove(output_model)
                    else:
                        shutil.rmtree(output_model)
                    print("删除完成")
                except Exception as e:
                    print(f"删除文件夹时出错: {e}")
                    return False
        
            # 创建b路径
            print(f"创建文件夹: {output_model}")
            os.makedirs(output_model, exist_ok=True)
            
            # 拷贝a路径下的所有内容到b路径
            print(f"正在从 '{self.original_model}' 拷贝文件到 '{output_model}'...")

            # 递归拷贝所有内容
            for item in os.listdir(self.original_model):
                src_item = os.path.join(self.original_model, item)
                dst_item = os.path.join(output_model, item)
                    
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                    print(f"拷贝文件: {item}")
                elif os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                    print(f"拷贝文件夹: {item}")
        
            print("操作完成！")
            return True
            
        except Exception as e:
            print(f"操作过程中出错: {e}")
            return False
