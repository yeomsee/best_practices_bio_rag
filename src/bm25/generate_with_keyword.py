from rank_bm25 import BM25Okapi
import csv
import json
import re
from tqdm import tqdm
from src.dpr.api_call import *

def load_file(retriever_name, exp_num, mode_name):
    """
    mode_list = ["limit_3", "no_limit"]
    """

    dir_name = f"/home/stv10121/RUS/{retriever_name}_result"
    file_name = f"{dir_name}/{exp_num}_{mode_name}.json"

    with open(file_name) as in_file:
        data = json.load(in_file)
    
    return data["data"]

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

def preprocess_text(text):
    pattern = r"^\d+\.\s*(.+)$"
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches

def _retrieve(query, doc_list, retriever, n=1):
    tokenized_query = query.lower().split(" ")

    return retriever.get_top_n(tokenized_query, doc_list, n)

def main():
    csv_fname = "./result_all_new_1125.csv"
    
    # load title and abstract
    _, doc_list = load_csv(csv_fname)

    # tokenize abstract
    tokenized_corpus = [doc.split(" ") for doc in doc_list]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # random sampling
    with open("/home/stv10121/RUS/queries_eng.txt", 'r', encoding='utf-8') as in_file:
        queries = in_file.readlines()
    queries = [query.replace('\n', "") if '\n' in query else query for query in queries]

    for id, query in enumerate(tqdm(queries)):
        # get keywords
        keywords = get_general_keywords(query)
        keywords = preprocess_text(keywords)
        keywords = [keyword.strip() for keyword in keywords]

        for mode in ["hyde", "rewrite"]:
            # file open and correction
            file_path = f"/home/stv10121/RUS/bm25_result/keyword_{mode}.json"

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                        result_dict = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                    result_dict = {"data": []}

            if mode == "hyde":
                # hyde
                hop_1_queries = [get_abstract_from_keyword(query, keyword) for keyword in keywords]
            else: # mode == "rewrite"
                # rewrite
                hop_1_queries = [get_query_from_keyword(query, keyword) for keyword in keywords]

            # 1-hop retrieval
            hop_1_papers = [_retrieve(hop_1_query, doc_list, bm25, n=1)[0] for hop_1_query in hop_1_queries]

            # 2-hop retrieval
            hop_2_queries = [(hop_1_papers[q_idx] + hop_1_queries[q_idx]) for q_idx in range(len(hop_1_queries))]
            hop_2_papers = [_retrieve(hop_2_query, doc_list, bm25, n=2) for hop_2_query in hop_2_queries]

            # removed duplicated documents
            hop_2_papers = [hop_2_papers[q_idx][1] if (hop_2_papers[q_idx][0] == hop_1_papers[q_idx]) else hop_2_papers[q_idx][0] for q_idx in range(len(hop_2_papers))]

            papers = [f"<p1>{hop_1_paper}</p1><p2>{hop_2_paper}</p2>" for hop_1_paper, hop_2_paper in zip(hop_1_papers, hop_2_papers)]

            # generate answer by LLM
            output = generate(query, papers, keywords)

            # save
            result_dict["data"].append(
                {
                    "id": id,
                    "query_org": query,
                    "results": [
                    {
                        "keyword": keywords[q_idx],
                        f"query_{mode}": hop_1_queries[q_idx],
                        "1_hop_paper": hop_1_papers[q_idx],
                        "2_hop_paper": hop_2_papers[q_idx]
                    } for q_idx in range(len(hop_1_queries))
                ],
                    "output": output
                })

            # save file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent="\t")

if __name__ == '__main__':
    main()