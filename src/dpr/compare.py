import json
from tqdm import tqdm
from api_call import *

def load_file(retriever_name, exp_num):
    mode_list = ["limit_3", "no_limit"]
    dir_name = f"/home/stv10121/RUS/{retriever_name}_result"

    file_names = [f"{dir_name}/{exp_num}_{mode}.json" for mode in mode_list]

    datas = []
    for file_name in file_names:
        with open(file_name) as in_file:
            data = json.load(in_file)
            datas.append(data)
    
    datas1, datas2 = datas
    return datas1["data"], datas2["data"]

if __name__ == "__main__":
    for exp in ["exp_1", "exp_2", "exp_3", "exp_4", "exp_5"]:
        with_limit, with_no_limit = load_file("DPR", exp)

        preference_results = {}
        for x, y in tqdm(zip(with_limit, with_no_limit)):
            query_org = x["query_org"]
            x_results, y_results = x["results"], y["results"]
            if (exp == "exp_2") or (exp == "exp_4"):
                x_rewritten_queries, y_rewritten_queries = [], []
                for x_result, y_result in zip(x_results, y_results):
                    x_rewritten_queries.append(x_result["rewrite_query"])
                    y_rewritten_queries.append(y_result["rewrite_query"])
                
                preference_results[query_org] = get_preference(query_org, x_rewritten_queries, y_rewritten_queries)
            else:
                x_virtual_abstracts, y_virtual_abstracts = [], []
                for x_result, y_result in zip(x_results, y_results):
                    x_virtual_abstracts.append(x_result["virtual_abstract"])
                    y_virtual_abstracts.append(y_result["virtual_abstract"])
                
                preference_results[query_org] = get_preference(query_org, x_virtual_abstracts, y_virtual_abstracts, type="expansion")
        
        json_file = f"/home/stv10121/RUS/DPR_result/{exp}_preference.json"
        with open(json_file, 'w', encoding='utf8') as out_file:
            json.dump(preference_results, out_file, indent="\t", ensure_ascii=False)