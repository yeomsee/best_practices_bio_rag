import csv
import json
import re
import numpy as np

from tqdm import tqdm

from src.dpr.api_call import *


def load_file(retriever_name, exp_num, mode_name):
    """
    mode_list = ["limit_3", "no_limit"]
    """

    dir_name = f"/home/stv10121/RUS/{retriever_name}_result"
    file_name = f"{dir_name}/{exp_num}_{mode_name}_mesh.json"

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


# tf-idf를 이용해 문서의 다양성 측정하는 함수
def diversity_cosine_sim(doc_list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(doc_list)
    cosine_sim_matrix = cosine_similarity(doc_vectors)

    upper_triangles_indices = np.tril_indices_from(cosine_sim_matrix, k=1) # k=0이면 대각선부터 시작, k=1이면 대각선위부터 시작해서 상삼각형 부분의 인덱스 반환
    average_cosine_sim = np.mean(cosine_sim_matrix[upper_triangles_indices])
    diversity_score = round(1-average_cosine_sim, 2)

    return diversity_score


if __name__ == '__main__':
    retriever_list = ["bm25"]
    mode_list = ["keyword"]
    exp_list = ["exp_4", "exp_5"]
    for retriever in retriever_list:
        for mode in mode_list:
            for exp in exp_list:
                result_dict = {"data": []}
                if retriever == "bm25":
                    json_file = f"/home/stv10121/RUS/{retriever}_result/{exp}_{mode}_mesh_test.json"
                else: # retriever == "DPR"
                    json_file = f"/home/stv10121/RUS/{retriever}_result/{exp}_{mode}_mesh_test.json"
                out_file = open(json_file, 'w', encoding='utf-8')

                # load data
                datas = load_file("bm25", exp, mode)
                print(f"\n##### {retriever}_{exp}_{mode} Retrieve/Generate ######")
                if exp == "exp_1":
                    for data in tqdm(datas):
                        query = data["query_org"]
                        results = data["results"]
                        
                        # retrieve/generate and evaluate
                        hop_1_papers = [result["1_hop_paper"] for result in results]
                        hop_1_outputs = [generate(query, hop_1_paper) for hop_1_paper in hop_1_papers]
                        hop_1_evals = [eval_insightfulness(query, hop_1_output) for hop_1_output in hop_1_outputs]

                        # save
                        result_dict["data"].append(
                            {
                                "query_org": query,
                                "results": [
                                    {
                                        "1_hop_paper": hop_1_papers[q_idx],
                                        "output": hop_1_outputs[q_idx],
                                        "insight_score": hop_1_evals[q_idx]
                                    } for q_idx in range(len(hop_1_outputs))
                                ]
                            }
                        )
                elif (exp == "exp_2") or (exp == "exp_4"):
                    for data in tqdm(datas):
                        query = data["query_org"]
                        results = data["results"]
                        
                        # retrieve and evaluate
                        rewrite_queries = [result["rewrite_query"] for result in results]

                        keywords = [result["keyword"] for result in results]
                        keywords_unique = list(set(keywords))

                        hop_1_papers = [result["1_hop_paper"] for result in results]
                        hop_1_papers_unique = list(set(hop_1_papers))

                        # diversity_org_ver
                        diversity_org = round(len(hop_1_papers_unique)/len(keywords_unique), 2)

                        # relevance_scores
                        relevance_scores = [int(result["relevance"]) for result in results]
                        relevance_score_avg = np.mean(relevance_scores)
                        relevance_score_avg = round(relevance_score_avg, 2)

                        # diversity_cos_ver
                        diversity_cos_org = diversity_cosine_sim(hop_1_papers)
                        diversity_cos_unique = diversity_cosine_sim(hop_1_papers_unique)

                        # save
                        result_dict["data"].append(
                            {
                                "query_org": query,
                                "results": [
                                    {
                                        "keyword": keywords[q_idx],
                                        "rewrite_query": rewrite_queries[q_idx],
                                        "1_hop_paper": hop_1_papers[q_idx],
                                        "relevance_score":relevance_scores[q_idx]
                                    } for q_idx in range(len(hop_1_papers))
                                ],
                                "total_documents_number": len(hop_1_papers),
                                "unique_documents_number": len(hop_1_papers_unique),
                                "relevance_score_avg": relevance_score_avg,
                                "diversity_org": diversity_org,
                                "diversity_cos_org": diversity_cos_org,
                                "diversity_cos_unique": diversity_cos_unique
                                }
                            )
                else: # (mode == "exp_3") or (mode == "exp_5")
                    for data in tqdm(datas):
                        query = data["query_org"]
                        results = data["results"]
                        
                        # retrieve and evaluate
                        virtual_abstracts = [result["virtual_abstract"] for result in results]

                        keywords = [result["keyword"] for result in results]
                        keywords_unique = list(set(keywords))

                        hop_1_papers = [result["1_hop_paper"] for result in results]
                        hop_1_papers_unique = list(set(hop_1_papers))

                        # diversity_org_ver
                        diversity_org = round(len(hop_1_papers_unique)/len(keywords_unique), 2)

                        # relevance_scores
                        relevance_scores = [int(result["relevance"]) for result in results]
                        relevance_score_avg = np.mean(relevance_scores)
                        relevance_score_avg = round(relevance_score_avg, 2)

                        # diversity_cos_ver
                        diversity_cos_org = diversity_cosine_sim(hop_1_papers)
                        diversity_cos_unique = diversity_cosine_sim(hop_1_papers_unique)

                        # save
                        result_dict["data"].append(
                            {"query_org": query,
                             "results": [
                                    {
                                        "keyword": keywords[q_idx],
                                        "virtual_abstract": virtual_abstracts[q_idx],
                                        "1_hop_paper": hop_1_papers[q_idx],
                                        "relevance_score":relevance_scores[q_idx]
                                    } for q_idx in range(len(hop_1_papers))
                                ],
                                "total_documents_number": len(hop_1_papers),
                                "unique_documents_number": len(hop_1_papers_unique),
                                "relevance_score_avg": relevance_score_avg,
                                "diversity_org": diversity_org,
                                "diversity_cos_org": diversity_cos_org,
                                "diversity_cos_unique": diversity_cos_unique
                                })

                json.dump(result_dict, out_file, indent='\t', ensure_ascii=False)
                out_file.close()