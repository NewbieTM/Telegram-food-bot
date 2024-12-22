import argparse
import os
import shutil
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



CHROMA_PATH = "../db/chroma"
DATA_PATH = "../db/data.txt"
BATCH_SIZE = 1000
MAX_WORKERS = 4


def get_embedding_function():
    """
    Инициализирует функцию генерации эмбеддингов с использованием модели HuggingFace.
    Использует GPU, если доступен.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используем устройство для эмбеддингов: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="DeepPavlov/rubert-base-cased-sentence-transformer",
        model_kwargs={"device": device}
    )
    return embeddings


def load_documents():
    """
    Загружает документы из текстового файла.
    Каждый продукт рассматривается как отдельный документ.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл данных не найден по пути: {DATA_PATH}")

    logger.info("Загрузка документов из файла...")
    document_loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = document_loader.load()



    def preprocess_text(text):
        text = text.lower()
        text = text.strip()
        return text

    split_docs = []
    for doc in documents:
        lines = doc.page_content.split('\n')
        for line in lines:
            line = preprocess_text(line)
            if line:
                split_docs.append(Document(page_content=line))

    logger.info(f"🔍 Количество загруженных документов: {len(split_docs)}")
    return split_docs


def split_documents(documents: list[Document]):
    """
    Разделяет документы на более мелкие чанки для лучшей векторизации.
    """
    logger.info("Разделение документов на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=40,
        length_function=len,
        # Для русского языка можно настроить разделители, если необходимо
    )
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"🔗 Количество чанков после разделения: {len(split_docs)}")
    return split_docs


def clear_database():
    """
    Очищает существующую базу данных Chroma.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info(f"🗑️ База данных по пути {CHROMA_PATH} была удалена.")
    else:
        logger.info(f"ℹ️ База данных по пути {CHROMA_PATH} не существует.")


def add_to_chroma(chunks: list[Document], embeddings, batch_size=BATCH_SIZE):
    """
    Добавляет чанки документов в базу данных Chroma и сохраняет её.
    Обрабатывает документы пакетами для повышения эффективности.
    """
    logger.info("Инициализация базы данных Chroma...")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    total_chunks = len(chunks)
    logger.info(f"Начало добавления {total_chunks} чанков в Chroma...")

    for i in tqdm(range(0, total_chunks, batch_size), desc="Добавление чанков"):
        batch = chunks[i:i + batch_size]
        try:
            db.add_documents(batch)
            logger.debug(f"Добавлено {i + len(batch)} из {total_chunks} чанков.")
        except Exception as e:
            logger.error(f"Ошибка при добавлении чанков с {i} по {i + len(batch)}: {e}")

    db.persist()
    logger.info(f"✅ Векторизованная база данных создана и сохранена по пути {CHROMA_PATH}.")


def process_batches(chunks, embeddings, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    """
    Обрабатывает и добавляет чанки документов в Chroma параллельно.
    """
    logger.info("Начало параллельной обработки чанков и добавления в Chroma...")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    total_chunks = len(chunks)
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            futures.append(executor.submit(db.add_documents, batch))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка чанков"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Ошибка при обработке батча: {e}")

    db.persist()
    logger.info(f"✅ Векторизованная база данных создана и сохранена по пути {CHROMA_PATH}.")


def main():
    """
    Основная функция, которая управляет процессом создания базы данных.
    """
    parser = argparse.ArgumentParser(description="Создание векторизованной базы данных.")
    parser.add_argument("--reset", action="store_true", help="Сбросить базу данных перед созданием.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Размер батча для обработки.")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS,
                        help="Количество потоков для параллельной обработки.")
    parser.add_argument("--use_parallel", action="store_true", help="Использовать параллельную обработку.")
    args = parser.parse_args()

    if args.reset:
        logger.info("✨ Очищение базы данных...")
        clear_database()

    documents = load_documents()
    #chunks = split_documents(documents)
    chunks = documents

    embeddings = get_embedding_function()

    if args.use_parallel:
        process_batches(chunks, embeddings, batch_size=args.batch_size, max_workers=args.max_workers)
    else:
        add_to_chroma(chunks, embeddings, batch_size=args.batch_size)

    logger.info("✅ База данных успешно создана.")


if __name__ == "__main__":
    main()
