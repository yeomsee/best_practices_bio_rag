
import csv
import json
import re

from tqdm import tqdm
from rank_bm25 import BM25Okapi

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


if __name__ == '__main__':
    csv_fname = "./result_all_new_1125.csv"
    
    # load title and abstract
    title_list, doc_list = load_csv(csv_fname)

    # tokenize abstract
    tokenized_corpus = [doc.split(" ") for doc in doc_list]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # random sampling
    with open("/home/stv10121/RUS/queries_eng_100.txt", 'r', encoding='utf-8') as in_file:
        queries = in_file.readlines()
    queries = [query.replace('\n', "") if '\n' in query else query for query in queries][:20]
    
    # exp_list = ["exp_1", "exp_2", "exp_3", "exp_4", "exp_5"]
    exp_list = ["exp_4", "exp_5"]
    for exp in exp_list:
        result_dict = {"data": []}
        json_file = f"/home/stv10121/RUS/bm25_result/{exp}_keyword_mesh.json"
        out_file = open(json_file, 'w', encoding='utf-8')

        for query in tqdm(queries):
            if exp == "exp_4":
                keywords = get_general_keywords(query)
                keywords = preprocess_text(keywords)

                hop_1_inputs = [get_query_from_keyword(query, keyword) for keyword in keywords]

                # retreive and evaluate
                hop_1_outputs, hop_1_evals = [], []
                for hop_1_input in hop_1_inputs:
                    tokenized_query = hop_1_input.split(" ")
                    hop_1_output = bm25.get_top_n(tokenized_query, doc_list, n=1)[0]
                    hop_1_outputs.append(hop_1_output)
                    
                    hop_1_eval = evaluate_retrieval(query, hop_1_output)
                    hop_1_eval = hop_1_eval.strip().split('\t')
                    hop_1_evals.append(hop_1_eval)

                # save
                result_dict["data"].append(
                    {"query_org": query,
                     "results": [
                        {
                            "keyword": keywords[q_idx],
                            "rewrite_query": hop_1_inputs[q_idx],
                            "1_hop_paper": hop_1_outputs[q_idx],
                            "relevance": hop_1_evals[q_idx][1]
                        } for q_idx in range(len(hop_1_inputs))
                    ]
                    })
            else:
                keywords = get_mesh_keywords(query)
                keywords = preprocess_text(keywords)

                hop_1_inputs = [get_abstract_from_keyword(query, keyword) for keyword in keywords]

                # retreive and evaluate
                hop_1_outputs, hop_1_evals = [], []
                for hop_1_input in hop_1_inputs:
                    tokenized_query = hop_1_input.split(" ")
                    hop_1_output = bm25.get_top_n(tokenized_query, doc_list, n=1)[0]
                    hop_1_outputs.append(hop_1_output)
                    
                    hop_1_eval = evaluate_retrieval(query, hop_1_output)
                    hop_1_eval = hop_1_eval.strip().split('\t')
                    hop_1_evals.append(hop_1_eval)

                # save
                result_dict["data"].append(
                    {"query_org": query,
                     "results": [
                        {
                            "keyword": keywords[q_idx],
                            "virtual_abstract": hop_1_inputs[q_idx],
                            "1_hop_paper": hop_1_outputs[q_idx],
                            "relevance": hop_1_evals[q_idx][1]
                        } for q_idx in range(len(hop_1_inputs))
                    ]
                    })
    
    # exp_list = ["exp_2"]
    # for exp in exp_list:
    #     result_dict = {"data": []}
    #     json_file = f"./bm25_result/{exp}_limit_3.json"
    #     out_file = open(json_file, 'w', encoding='utf-8')

    #     print(f"\n##### {exp} query loading #####")
    #     for query in tqdm(queries):
    #         if exp == "exp_1":  # naive
    #             hop_1_inputs = [query]

    #         elif exp == "exp_2":  # aspect에 따른 query rewrite (aspect 사전 정의 X)
    #             hop_1_inputs = get_query_from_aspect(query, aspect=None, num=3)
    #             hop_1_inputs = preprocess_text(hop_1_inputs)

    #             # Check the scores of the rewrite-queries
    #             scores = get_queries_fitness_and_diversity(query, hop_1_inputs)
    #             fitness_score, diversity_score = scores.strip().split('\t')

    #         elif exp == "exp_3":  # naive query expansion
    #             hop_1_inputs = get_virtual_abstract(query, aspect=None, num=3)
    #             hop_1_inputs = preprocess_text(hop_1_inputs)

    #             # Check the scores of the abstracts
    #             scores = get_abstracts_fitness_and_diversity(query, hop_1_inputs)
    #             fitness_score, diversity_score = scores.strip().split('\t')

    #         elif exp == "exp_4":  # aspect 추출 후 query+aspect -> aspect에 따른 질문 생성
    #             # get_aspect(query, num=None)
    #             aspects = get_aspect(query, num=3)
    #             aspects = preprocess_text(aspects)

    #             # query+aspect -> aspect에 따른 질문 생성
    #             hop_1_inputs = [get_query_from_aspect(query, aspect) for aspect in aspects]

    #             # Check the scores of the abstracts
    #             scores = get_queries_fitness_and_diversity(query, hop_1_inputs)
    #             fitness_score, diversity_score = scores.strip().split('\t')

    #         else:  # "exp_5": aspect 추출 후 query+aspect -> aspect에 따른 새로운 abstract 생성
    #             # extract aspect
    #             aspects = get_aspect(query, num=3)
    #             aspects = preprocess_text(aspects)

    #             # from extracted aspects, 'aspect + query' -> new abstract
    #             hop_1_inputs = [get_virtual_abstract(query, aspect) for aspect in aspects]

    #             # Check the scores of the abstracts
    #             scores = get_abstracts_fitness_and_diversity(query, hop_1_inputs)
    #             fitness_score, diversity_score = scores.strip().split('\t')

    #         print(f"\n##### {exp} 1-hop retrieval #####")
    #         hop_1_outputs = []
    #         for hop_1_input in hop_1_inputs:
    #             tokenized_query = hop_1_input.split(" ")
    #             hop_1_output = bm25.get_top_n(tokenized_query, doc_list, n=1)[0]
    #             hop_1_outputs.append(hop_1_output)

    #         print(f"\n##### {exp} evaluation #####")
    #         hop_1_evals = []
    #         for hop_1_output in hop_1_outputs:
    #             hop_1_eval = evaluate_retrieval(query, hop_1_output)
    #             hop_1_eval = hop_1_eval.strip().split('\t')
    #             hop_1_evals.append(hop_1_eval)

    #         # save exp results 
    #         if (exp == "exp_1"):
    #             result_dict["data"].append(
    #                 {"query_org": query,
    #                  "results": [
    #                      {
    #                          "1_hop_paper": hop_1_outputs[q_idx],
    #                          "Factfulness": hop_1_evals[q_idx][0],
    #                             "Relevance": hop_1_evals[q_idx][1],
    #                             "Clarity": hop_1_evals[q_idx][2],
    #                             "Insight": hop_1_evals[q_idx][3]
    #                      } for q_idx in range(3)
    #                  ]
    #                  })
    #         elif (exp == "exp_2"):
    #             result_dict["data"].append(
    #                 {"query_org": query,
    #                  "results": [
    #                      {
    #                          "rewrite_query": hop_1_inputs[q_idx],
    #                          "1_hop_paper": hop_1_outputs[q_idx],
    #                          "Factfulness": hop_1_evals[q_idx][0],
    #                             "Relevance": hop_1_evals[q_idx][1],
    #                             "Clarity": hop_1_evals[q_idx][2],
    #                             "Insight": hop_1_evals[q_idx][3]
    #                      } for q_idx in range(3)
    #                  ],
    #                 "rewrite_diversity_score": diversity_score,
    #                 "rewrite_fitness_score": fitness_score,
    #                  })
    #         elif (exp == "exp_3"):
    #             result_dict["data"].append(
    #                 {"query_org": query,
    #                  "results": [
    #                      {
    #                          "virtual_abstract": hop_1_inputs[q_idx],
    #                          "1_hop_paper": hop_1_outputs[q_idx],
    #                          "Factfulness": hop_1_evals[q_idx][0],
    #                             "Relevance": hop_1_evals[q_idx][1],
    #                             "Clarity": hop_1_evals[q_idx][2],
    #                             "Insight": hop_1_evals[q_idx][3]
    #                      } for q_idx in range(3)
    #                  ],
    #                 "expansion_diversity_score": diversity_score,
    #                 "expansion_fitness_score": fitness_score,
    #                  })
    #         elif (exp == "exp_4"):
    #             result_dict["data"].append(
    #                 {"query_org": query,
    #                  "results": [
    #                      {
    #                          "aspect": aspects[q_idx],
    #                          "rewrite_query": hop_1_inputs[q_idx],
    #                          "1_hop_paper": hop_1_outputs[q_idx],
    #                          "Factfulness": hop_1_evals[q_idx][0],
    #                             "Relevance": hop_1_evals[q_idx][1],
    #                             "Clarity": hop_1_evals[q_idx][2],
    #                             "Insight": hop_1_evals[q_idx][3]
    #                      } for q_idx in range(3)
    #                  ],
    #                  "rewrite_diversity_score": diversity_score,
    #                  "rewrite_fitness_score": fitness_score
    #                  })
    #         else:  # (exp == "exp_5")
    #             result_dict["data"].append(
    #                 {"query_org": query,
    #                  "results": [
    #                      {
    #                          "aspect": aspects[q_idx],
    #                          "virtual_abstract": hop_1_inputs[q_idx],
    #                          "1_hop_paper": hop_1_outputs[q_idx],
    #                          "Factfulness": hop_1_evals[q_idx][0],
    #                             "Relevance": hop_1_evals[q_idx][1],
    #                             "Clarity": hop_1_evals[q_idx][2],
    #                             "Insight": hop_1_evals[q_idx][3]
    #                      } for q_idx in range(3)
    #                  ],
    #                 "expansion_diversity_score": diversity_score,
    #                 "expansion_fitness_score": fitness_score,
    #                 })
             
        json.dump(result_dict, out_file, indent='\t', ensure_ascii=False)
        out_file.close()