import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import kagglehub


project_cwd = Path.cwd()
dataset_dir = project_cwd / "dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
if path:
    src = Path(path).expanduser().resolve()
    if src.exists() and src.is_dir():
        for item in src.iterdir():
            dest = dataset_dir / item.name
            if dest.exists():
                print(f"Skipping existing: {dest}")
                continue
            try:
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
                print(f"Copied {item} -> {dest}")
            except Exception as e:
                print(f"Failed to copy {item}: {e}")
        print("Data copied from kagglehub download to project 'dataset' directory.")
    else:
        print(f"Downloaded path '{src}' does not exist or is not a directory.")




