import json
import random

from tqdm import tqdm

def sampling(datas, sample_size=50, seed=42):
    random.seed(seed)
    sampled_datas = random.sample(datas, sample_size)
    return sampled_datas


def main():
    retrievers = ["bm25", "dpr"]
    modes = ["hyde"]
    
    for retriever in retrievers:
        for mode in modes:
            # random sampling
            with open(f"/home/stv10121/RUS/{retriever}_result/keyword_mesh_{mode}.json", 'r', encoding='utf-8') as in_file:
                datas = json.load(in_file)["data"]
            datas = sampling(datas)

            result_dict = {"data": []}
            for data in tqdm(datas):
                id = data["id"]
                query_org = data["query_org"]
                output = data["output"]
                results = data["results"]

                hop_1_queries = [result[f"query_{mode}"] for result in results]
                hop_1_papers = [result["1_hop_paper"] for result in results]
                hop_2_papers = [result["2_hop_paper"] for result in results]

                # save
                result_dict["data"].append(
                {
                    "id": id,
                    "query_org": query_org,
                    "results": [
                    {
                        f"query_{mode}": hop_1_queries[q_idx],
                        "1_hop_paper": hop_1_papers[q_idx],
                        "2_hop_paper": hop_2_papers[q_idx]
                    } for q_idx in range(len(hop_1_queries))
                    ],
                    "output": output,
                    "factuality": "",
                    "diversity": "",
                    "clarity": "",
                    "insightfulness": ""  
                })

            out_file_path = f"data/{retriever}_keyword_mesh_{mode}.json"
            with open(out_file_path, 'w', encoding='utf-8') as out_file:
                json.dump(result_dict, out_file, indent="\t", ensure_ascii=False)


if __name__ == '__main__':
    main()