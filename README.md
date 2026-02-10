# Searching for the Best Practices of Query Reformulation in Bio-Insight Generation
This repository contains the source code and essential experimental data for the research paper: "Searching for the Best Practices of Query Reformulation in Bio-Insight Generation" (Submitted to BMC Bioinformatics, 2026).

## 📌 Project Overview
Our research explores the optimal strategies for Query Reformulation to enhance retrieval performance in Bio-Insight generation tasks. Focusing on the Alzheimer's disease domain, we evaluate various reformulation techniques using user-centered queries collected from real-world biomedical research contexts.

## 📊 Data Resources
To comply with GitHub's file size limits and ensure high-speed access to large datasets, we have distributed our resources as follows:

### 1. GitHub (This Repository)
Contains source code, environment configurations, and lightweight query data.

data/queries_eng.json: Experimental queries reflecting real-world information-seeking behaviors.

src/: Implementation of the retrieval pipeline (BM25 and DPR).

### 2. Hugging Face (Large-scale Assets)
Contains the heavy document corpus and pre-built vector indices.

Hugging Face Hub: stv10121/pubmed_alzheimer_kb

documents.tsv: External knowledge base consisting of PubMed abstracts (2015–2024).

index/: Pre-computed DPR vector indices for immediate reproduction of our results.
