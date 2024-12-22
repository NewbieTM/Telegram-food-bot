from langchain_huggingface import HuggingFaceEmbeddings
import argparse
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma


def get_embedding_function():
    """
    Инициализирует функцию генерации эмбеддингов с использованием модели HuggingFace.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sberbank-ai/sbert_large_nlu_ru")
    return embeddings


def load_documents():
    """
    Загружает документы из текстового файла.
    Вариант 1: Каждый продукт как отдельный документ (без разделения на чанки).
    Вариант 2: Все продукты объединены в один документ, затем разделены на чанки.
    Выберите подходящий вариант.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл данных не найден по пути: {DATA_PATH}")

    document_loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = document_loader.load()

    # Вариант 1: Каждый продукт как отдельный документ
    split_docs = []
    for doc in documents:
        lines = doc.page_content.split('\n')
        for line in lines:
            line = line.strip()
            if line:  # Пропускаем пустые строки
                split_docs.append(Document(page_content=line))
    return split_docs

    # Вариант 2: Все продукты в одном документе
    # content = "\n".join([doc.page_content for doc in documents])
    # return [Document(page_content=content)]


def split_documents(documents: list[Document]):
    """
    Разделяет документы на более мелкие чанки для лучшей векторизации.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
        length_function=len,
        # Для русского языка можно оставить is_separator_regex=False или настроить по необходимости
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    """
    Добавляет чанки документов в базу данных Chroma и сохраняет её.
    """
    embeddings = get_embedding_function()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"✅ Векторизованная база данных создана и сохранена по пути {CHROMA_PATH}.")


def clear_database():
    """
    Очищает существующую базу данных Chroma.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"🗑️ База данных по пути {CHROMA_PATH} была удалена.")
    else:
        print(f"ℹ️ База данных по пути {CHROMA_PATH} не существует.")


CHROMA_PATH = "../db/chroma"
DATA_PATH = "../db/data.txt"


def main():
    """
    Основная функция, которая управляет процессом создания базы данных.
    """
    parser = argparse.ArgumentParser(description="Создание векторизованной базы данных.")
    parser.add_argument("--reset", action="store_true", help="Сбросить базу данных перед созданием.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Очищение базы данных...")
        clear_database()

    print("📄 Загрузка документов...")
    documents = load_documents()
    print(f"🔍 Количество загруженных документов: {len(documents)}")

    print("✂️ Разделение документов на чанки...")
    chunks = split_documents(documents)
    print(f"🔗 Количество чанков после разделения: {len(chunks)}")

    print("🔗 Добавление чанков в Chroma и создание базы данных...")
    add_to_chroma(chunks)
    print("✅ База данных успешно создана.")


if __name__ == "__main__":
    main()