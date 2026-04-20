import subprocess
import os
import subprocess
import sys

base_dir = "Assinment_2(Mechintrap)"

scripts = [
    "inference.py",
    "rank.py",
    "robust_quant.py",
    "vae.py",
    "robust_vae.py"
]

for script in scripts:
    script_path = os.path.join(base_dir, script)

    print(f"\nRunning {script_path}")
    
    try:
        subprocess.run(
            ["python", script_path],
            check=True
        )
        print(f"{script} finished successfully.")

    except subprocess.CalledProcessError:
        print(f"{script} failed. Skipping...\n")
        continue

    except FileNotFoundError:
        print(f"{script} not found. Skipping...\n")
        continue

print("\nAll scripts attempted.")