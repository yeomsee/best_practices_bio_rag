from src.data_utils import load_csv, save_json_file


def main():
    paper_titles, paper_abstracts = load_csv("result_all_new_1125.csv")
    results = []
    for idx, (paper_title, paper_abstract) in enumerate(zip(paper_titles, paper_abstracts)):
        results.append(
            {
                'id': idx,
                'title': paper_title,
                'abstract': paper_abstract
            }
        )
    
    # save to jsonl
    output_file_path = "data/documents.jsonl"
    save_json_file(output_file_path, results)
    print(f"Saved {len(results)} documents to {output_file_path}")

if __name__ == "__main__":
    main()