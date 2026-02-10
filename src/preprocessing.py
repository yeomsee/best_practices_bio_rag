import csv
import os


from tqdm import tqdm

titles = []

def normalize_sentence(sentence):
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("棺", "β")
    sentence = sentence.replace("짹", "±")
    sentence = sentence.replace("혻", "")
    sentence = sentence.replace("慣", "α")
    sentence = sentence.replace("琯","ε")
    sentence = sentence.replace("?셲", "'s")
    sentence = sentence.replace("諭늖", "–me")
    sentence = sentence.replace("뀼", "'s")
    sentence = sentence.replace("셲", "'s")
    sentence = sentence.replace("汝", "β")
    sentence = sentence.replace("겖", "β")
    sentence = sentence.replace("棺-amyloid", "β-amyloid")
    sentence = sentence.replace("a棺", "aβ")
    
    return sentence


def load_csv(f_name):
    global titles
    result = []
    with open(f_name, 'rt', encoding='cp949', errors='ignore') as infile:
        data = csv.reader(infile)

        for idx, line in enumerate(data):
            if idx == 0:
                continue
            assert len(line) == 3
            if "No abstract available" in line[2]:
                continue
            if line[0] not in titles:
                titles.append(line[0])
                result.append([normalize_sentence(line[0]).lower(), line[1], normalize_sentence(line[2]).lower()])
    return result


def merge_csv(tsv_f_name, csv_f_name, dir_list):
    '''
    tsv - DPR용
    csv - bm25용
    '''
    tsv_file = open(tsv_f_name, 'w', encoding='utf-8', newline='')
    tw = csv.writer(tsv_file, delimiter='\t')
    tw.writerow(['id', 'title', 'text'])

    csv_file = open(csv_f_name, 'w', newline='')
    tw2 = csv.writer(csv_file)
    
    idx = 0
    for d in dir_list:
        f_list = os.listdir(d)
        for f_name in tqdm(f_list):
            datas = load_csv(os.path.join(d, f_name))
            for data in datas:
                data = [e.lower() for e in data]
                tw.writerow([idx, data[0], data[2]])
                tw2.writerow(data)
                idx+=1
    tsv_file.close()
    csv_file.close()


if __name__ == "__main__":
    tsv_f_name = "./result_all_new_1125.tsv"
    csv_f_name = "./result_all_new_1125.csv"
    
    dir_list = ["./papers/~2024", "./papers/240808"]

    merge_csv(tsv_f_name, csv_f_name, dir_list)