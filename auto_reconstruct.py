
import os
import subprocess

def run(cmd):
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)#stdout=subprocess.STDOUT if debug else subprocess.DEVNULL)
        p.wait()
    except KeyboardInterrupt:
        p.terminate()

path = "D:/deep_compression_old/eval"
python_exe = "C:/Users/Lenny/anaconda3/envs/deep-comp/python.exe"

data_path = "C:/Users/Lenny/deep_compression/ext/data"
eval_split_file_path = "C:/Users/Lenny/deep_compression/ext/splits/sv2_planes_test.json"
train_split_file_path = "C:/Users/Lenny/deep_compression/ext/splits/sv2_planes_train_500.json"

for dir in os.listdir(path):
    # using skip, so if everything is reconstructed already, nothing happens
    # reconstruct eval
    cmd = f"{python_exe} reconstruct.py --split {eval_split_file_path} -d {data_path} -e {os.path.join(path, dir)} -c 2000 --skip"
    run(cmd)
    # reconstruct train
    cmd = f"{python_exe} reconstruct.py --split {train_split_file_path} -d {data_path} -e {os.path.join(path, dir)} -c 2000 --skip"
    run(cmd)
    if not os.path.exists(os.path.join(path, dir, "Evaluation", "2000", "chamfer.csv")):
        # evaluate eval
        print("eval eval")
        cmd = f"{python_exe} evaluate.py --split {eval_split_file_path} -d {data_path} -e {os.path.join(path, dir)} -c 2000"
        run(cmd)
    if not os.path.exists(os.path.join(path, dir, "Evaluation", "2000", "chamfer_train.csv")):
        # evaluate train
        print("eval train")
        cmd = f"{python_exe} evaluate.py --split {train_split_file_path} -d {data_path} -e {os.path.join(path, dir)} -c 2000"
        run(cmd)