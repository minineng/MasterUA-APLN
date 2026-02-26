import os
import json
from typing import Dict
import pandas as pd

def clean_duplicated_images(folder: str):
    seen_hashes = set()
    removed_file_list = []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                filepath = os.path.join(folder, filename)
                with open(filepath, "rb") as f:
                    file_hash = hash(f.read())
                    if file_hash in seen_hashes:
                        os.remove(filepath)
                        removed_file_list.append(filename)
                    else:
                        seen_hashes.add(file_hash)

def write_json(data:Dict, filename:str):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(filename:str) -> Dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv(filename:str) -> Dict:
    return pd.read_csv(filename)

def write_csv(df, filename:str):
    df.to_csv(filename, index=False, encoding="utf-8")

def read_txt(filename:str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()
    
def write_txt(content:str, filename:str):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

def write_processed(data:Dict, filename:str):
    write_json(data, f"{filename}.json")
    df = pd.json_normalize(data, sep='_')
    write_csv(df, f"{filename}.csv")