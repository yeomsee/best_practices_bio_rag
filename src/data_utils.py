import csv
import json

from tqdm import tqdm

def load_csv(f_name):
    paper_titles = []
    paper_abstract = []
    with open(f_name, 'rt', encoding='cp949', errors='ignore') as infile:
        data = csv.reader(infile)

        for idx, line in tqdm(enumerate(data)):
            if idx == 0:
                continue
            assert len(line) == 3
            paper_titles.append(line[0].lower())
            paper_abstract.append(line[2].lower())
    return paper_titles, paper_abstract

def load_json_file(file_name):
    if file_name.endswith(".json"):
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_name.endswith(".jsonl"):
        with open(file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    return data

def load_txt_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data

def save_json_file(file_name, data):
    if file_name.endswith(".json"):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif file_name.endswith(".jsonl"):
        with open(file_name, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Invalid file extension: {file_name}")