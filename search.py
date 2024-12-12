import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")
CHROMA_PATH = "./db"


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name = "./model/all-MiniLM-L6-v2/",
        model_kwargs={"device": "cpu"},
    )
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=huggingface_embeddings)

   
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=1)
    final_data="\n\n---\n\n".join([doc.page_content for doc, _score in results])
    final_data_with_score=[_score for doc, _score in results]
    final_data_with_meta=[doc.metadata for doc, _score in results]
    print("==================================")
    print("==================================")
    print(final_data)
    print(final_data_with_score)
    print(final_data_with_meta)


if __name__ == "__main__":
    main()

# python3.11 -m venv newpython
# source newpython/bin/activate 
# python3 ingest.py 
# source deactivate