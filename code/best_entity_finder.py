import re
import sqlite3
from pymorphy3 import MorphAnalyzer
import pandas as pd
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

morph = MorphAnalyzer()

@lru_cache(maxsize=10000)
def cached_parse(word):
    return morph.parse(word)[0].normal_form

def clean_non_alphanumeric(word):
    return re.sub(r'[^a-zA-Zа-яА-Я]', '', word)

# Предобработка текста
def preprocess_text(text):
    text = re.sub(r'[0-9]+[,.]?[0-9]*\s?(кКал|г|грамм|мг|калорий|литров|мл|%|/100)', '', text)
    text = text.lower()
    text = ' '.join(clean_non_alphanumeric(word) for word in text.split())
    return text

def extract_morphemes(text):
    words = text.split()
    return {cached_parse(word) for word in words if len(word) > 2}

def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name};"
    data = pd.read_sql_query(query, conn, index_col='id')
    conn.close()
    data = data.drop_duplicates()

    data['original_text'] = data.apply(lambda row: ' '.join(row.astype(str)), axis=1)

    data['processed_text'] = data['original_text'].apply(preprocess_text)
    return data['processed_text'], data['original_text']

# Функция поиска наиболее похожих строк
def find_top_matches(prompt, processed_texts, original_texts, top_n=50):
    processed_prompt = preprocess_text(prompt)

    vectorizer = TfidfVectorizer(tokenizer=extract_morphemes)  # Используем морфемы как токены
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    query_vector = vectorizer.transform([processed_prompt])


    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()


    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_matches = [(cosine_similarities[i], original_texts.iloc[i]) for i in top_indices]

    return top_matches


if __name__ == "__main__":
    db_path = r'../db/products.db'
    table_name = "products"

    print("Загрузка данных из базы...")
    processed_texts, original_texts = load_data(db_path, table_name)

    prompt = 'Что вкусного можно приготовить из свинины, овощей и грибов?'

    print("Ищем совпадения...")
    top_50_matches = find_top_matches(prompt, processed_texts, original_texts, top_n=50)

    for idx, (similarity, text) in enumerate(top_50_matches, start=1):
        print(f"{idx}. Схожесть: {similarity:.2f}, Продукт: {text}")
