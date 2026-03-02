import os
import sys
import subprocess

mu_effs = [0.1, 0.5, 1.0, 0.05]
alphas = [1.0, 1.5, 2.0, 2.5, 3.0]

script_dir = os.path.dirname(os.path.abspath(__file__))

for run_id in range(10):
    for alpha in alphas:
        for mu_eff in mu_effs:
            subprocess.run([
                sys.executable, os.path.join(script_dir, "jam.py"),
                "--mu_eff", str(mu_eff),
                "--aspect_ratio", str(alpha),
                "--run_id", str(run_id),
            ], check=True)

            run_data_root = f'/home/mmccraw/dev/data/26-01-01/grant/specific-heat/mu-{mu_eff}-alpha-{alpha}/{run_id}'
            for step in sorted(os.listdir(run_data_root)):
                step_dir = os.path.join(run_data_root, step)
                if not os.path.isdir(step_dir):
                    continue
                jam_input = os.path.join(step_dir, 'jamming')
                cv_output = os.path.join(step_dir, 'cv')
                os.makedirs(cv_output, exist_ok=True)

                subprocess.run([
                    sys.executable, os.path.join(script_dir, "run-cv.py"),
                    "--input_root", jam_input,
                    "--output_root", cv_output,
                ], check=True)
