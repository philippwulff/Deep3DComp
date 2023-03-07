import json
import os
import subprocess

import deep_sdf.workspace as ws
import deep_sdf


DIR = "/home/shared/deepsdfcomp/architecture_comparisons/double_nls"
ARCHITECTURES = [("baseline", "/home/shared/deepsdfcomp/searches/double_nonlinearity/baseline"),
                 ("siren", "/home/shared/deepsdfcomp/searches/double_nonlinearity/all_latentsize=200_width=256_lr=5e-4_int=150"),
                 ("line", "/home/shared/deepsdfcomp/searches/double_nonlinearity/line"),
                 ("plane", "/home/shared/deepsdfcomp/searches/double_nonlinearity/plane")]
NUM_RECON_ITERS = 300
if __name__ == "__main__":
    for architecture in ARCHITECTURES:
        arch_name, arch_exp_dir = architecture
        
        cmd = f"python /home/freissmuth/deepsdf/reconstruct.py"
        cmd += f"--experiment {arch_exp_dir}"
        cmd += f"--data_source /home/shared/deepsdfcomp/data"
        cmd += f"--split {os.path.join(DIR, 'split.json')}"
        cmd += f"--iters {NUM_RECON_ITERS}"
        
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
        
    






    