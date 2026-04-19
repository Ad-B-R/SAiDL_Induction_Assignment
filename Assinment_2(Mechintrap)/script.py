import subprocess

scripts = [
    "inference.py",
    "rank.py",
    "robust_quant.py"
    "vae.py"
]

for script in scripts:
    print(f"\nRunning {script}")
    
    try:
        result = subprocess.run(
            ["python", script],
            check=True
        )
        print(f"{script} finished successfully.")

    except subprocess.CalledProcessError:
        print(f"{script} failed. Skipping...\n")
        continue

print("\nAll scripts attempted.")