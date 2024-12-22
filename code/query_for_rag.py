import argparse
from langchain_community.vectorstores import Chroma
from vectorize_faster_db import get_embedding_function

CHROMA_PATH = "../db/chroma"

def main():
    # Создание интерфейса командной строки
    parser = argparse.ArgumentParser(description="Поиск топ-25 наиболее релевантных строк по запросу.")
    parser.add_argument("query_text", type=str, nargs='?', help="Текст запроса.")
    args = parser.parse_args()

    if args.query_text:
        query_text = args.query_text
    else:
        query_text = input()

    query_text = query_text.lower().strip()

    query_rag(query_text)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=25)

    if not results:
        print("Нет результатов для данного запроса.")
        return

    print("\nТоп-25 релевантных строк:")
    for idx, (doc, score) in enumerate(results, start=1):
        full_text = doc.metadata.get("full_text", doc.page_content)
        print(f"{idx}. {full_text} (Score: {score:.4f})")

if __name__ == "__main__":
    main()
