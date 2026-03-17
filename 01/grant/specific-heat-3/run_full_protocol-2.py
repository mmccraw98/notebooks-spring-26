import os
import sys
import subprocess

mu_effs = [0.1, 0.5, 1.0, 0.05]

run_id = 0
alpha = 1.0
step = 4

script_dir = os.path.dirname(os.path.abspath(__file__))

for mu_eff in mu_effs:
    run_data_root = f'/home/mmccraw/dev/data/26-01-01/grant/specific-heat/mu-{mu_eff}-alpha-{alpha}/{run_id}'
    step_dir = os.path.join(run_data_root, str(step))
    if not os.path.isdir(step_dir):
        continue
    jam_input = os.path.join(step_dir, 'jamming')
    cv_output = os.path.join(step_dir, 'cv-2-low_temp')
    os.makedirs(cv_output, exist_ok=True)

    subprocess.run([
        sys.executable, os.path.join(script_dir, "run-cv-2.py"),
        "--input_root", jam_input,
        "--output_root", cv_output,
    ], check=True)
