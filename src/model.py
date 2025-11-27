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
    
    def continue_train_task(self, task_name, trainer, evaluater=None):
        self.task_name = task_name
        self.trainer = trainer
        self.evaluater = evaluater

    def train(self, epoch):
        self.trainer.get_source(self)
        self.trainer.train(epoch)

    def new_copy_for_original_model(self, output_model):
        self.output_model = output_model
        # Create a new training task by making a copy of the original model to avoid contamination
        self.output_model = output_model
        try:
            # Check if the source path exists
            if not os.path.exists(self.original_model):
                print(f"Error: The source folder '{self.original_model}' does not exist")
                return False
            
            # Check if the destination path exists
            if os.path.exists(output_model):
                print(f"The target folder '{output_model}' already exists. Deleting it...")
                try:
                    # Delete the destination path (including all files and subfolders)
                    if os.path.isfile(output_model):
                        os.remove(output_model)
                    else:
                        shutil.rmtree(output_model)
                    print("Deletion completed")
                except Exception as e:
                    print(f"Error occurred while deleting the folder: {e}")
                    return False
        
            # Create the destination path
            print(f"Creating folder: {output_model}")
            os.makedirs(output_model, exist_ok=True)
            
            # Copy all contents from the source path to the destination path
            print(f"Copying files from '{self.original_model}' to '{output_model}'...")

            # Recursively copy all contents
            for item in os.listdir(self.original_model):
                src_item = os.path.join(self.original_model, item)
                dst_item = os.path.join(output_model, item)
                    
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                    print(f"Copied file: {item}")
                elif os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                    print(f"Copied folder: {item}")
        
            print("Operation completed!")
            return True
            
        except Exception as e:
            print(f"An error occurred during the operation: {e}")
            return False
