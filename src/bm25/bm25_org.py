from rank_bm25 import BM25Okapi
import csv
import os
import json
from tqdm import tqdm
from dpr.api_call import get_virtual_abstract, get_discriminator, get_query_aspect_abstract_summary

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


if __name__ == '__main__':
    csv_fname = "./result_all_new_1125.csv"
    
    # load title and abstract
    title_list, corpus = load_csv(csv_fname)

    # tokenize abstract
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    queries = [
        "Please explain the mechanism through which exercise delays the progression of Alzheimer's disease.",
        "Describe the changes occurring in organs other than the brain in Alzheimer's disease patients",
        "Are there any studies on patients or experimental animals showing recovery after Alzheimer's disease?",
        "Explain the mechanism by which Tip60 overexpression rescues APP-mediated traffic jam.",
        "Explain the role of Def8 in the autophagy pathway and its regulation of the onset mechanism of Alzheimer's disease.",
        "Please explain possible mechanisms by which fatty acid binding protein may be involved in Alzheimer's disease.",
        "What research has been done on the association between Alzheimer's disease and lifespan?",
        "In cases of Alzheimer's disease, is there a genetic link associated with reduced lifespan?"
    ]
    
    mode_list = ["vanilla", "hyde_rewrite", "hyde_expansion"]
    for mode in mode_list:
        result_dict = {"data": []}
    
        # file to write
        json_file = f"./bm25_result/bm25_{mode}.json"
        output_file = open(json_file, 'w', encoding='utf8')

        if mode == "vanilla":
            for query in tqdm(queries):
                tokenized_query = query.lower().split(" ")
                top_10_paper = bm25.get_top_n(tokenized_query, corpus, n=10)
            
                # just retrieve top-1 paper
                for paper in top_10_paper:
                    if get_discriminator(query, paper):
                        hop_1_paper = paper
                        break

                result_dict["data"].append(
                    {"query": query,
                    "hop_1_paper": hop_1_paper})
        else: # hyde_rewrite, hyde_expansion
            for query in tqdm(queries):
                # HyDE (generate virtual abstract) for 1-hop
                new_query = get_virtual_abstract(query, mode)
                tokenized_query = new_query.lower().split(" ")

                # Retrieval (1-hop)
                top_10_paper = bm25.get_top_n(tokenized_query, corpus, n=10)
                for paper in top_10_paper:
                    # Discriminate
                    if get_discriminator(query, paper):
                        hop_1_paper = paper
                        break

                result_dict["data"].append(
                    {"query": query,
                     f"{mode}_query": new_query,
                     "hop_1_paper": hop_1_paper})
            
        json.dump(result_dict, output_file, indent="\t", ensure_ascii=False)
        output_file.close()
    
        # HyDE for 2-hop
        hop_1_summary = get_query_aspect_abstract_summary(query, hop_1_paper)

        # Regenerate query (query + 1-hop summary)
        tokenized_summary = tokenized_query + hop_1_summary.lower().split(" ")

        # Retrieval (2-hop)
        hop_2_papers = bm25.get_top_n(tokenized_summary, corpus, n=2)

        # 중복 제거
        if hop_2_papers[0] == hop_1_paper:
            hop_2_paper = hop_2_papers[1]
        else:
            hop_2_paper = hop_2_papers[0]

        result_dict["data"].append(
            {"question": query, "expanded_query": expanded_query,
                "1_hop_paper": hop_1_paper, "2-hop_expanded_query": hop_1_summary,
                "2_hop_paper": hop_2_paper})