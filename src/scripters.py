import subprocess
import threading

class Evaluater:
    def __init__(self, model_path, name, eval_type="None"):
        self.model_path = model_path
        self.name = name
        self.eval_type = eval_type

    def get_parameters(self, 
        eval_batch_size="64",
        eval_output_length="256", 
        eval_input_truncate_length="256",
        base_url = "your_base_url",
        api_key = "your_api_key"
        ):

        self.eval_batch_size = eval_batch_size
        self.eval_output_length = eval_output_length
        self.eval_input_truncate_length = eval_input_truncate_length
        self.base_url = base_url
        self.api_key = api_key

    def eval(self):
        cmd = [
            'python', '-u', 'src/scripts/eval.py',
            '--model_path', self.model_path,
            '--name', self.name,
            '--eval_type', self.eval_type,
            '--eval_batch_size', self.eval_batch_size,
            '--eval_output_length', self.eval_output_length,
            '--eval_input_truncate_length', self.eval_input_truncate_length,
            '--base_url', self.base_url,
            '--api_key', self.api_key
        ]
        cmd = [str(arg) if not isinstance(arg, (str, bytes)) else arg for arg in cmd] 
        print("Start Eval...")
        run_scripter(cmd)

def run_scripter(cmd):
    process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1, 
            universal_newlines=True
        )
        
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "STDOUT"))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "STDERR"))

    stdout_thread.start()
    stderr_thread.start()
    process.wait()

def read_output(stream, prefix):
    for line in iter(stream.readline, ''):
        print(f"{prefix}: {line}", end='')
    stream.close()