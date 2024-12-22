import re
from pymorphy3 import MorphAnalyzer
from functools import lru_cache
import random

morph = MorphAnalyzer()


@lru_cache(maxsize=10000)
def cached_parse(word):
    return morph.parse(word)[0].normal_form


def clean_non_alphanumeric(word):
    return re.sub(r'[^a-zA-Zа-яА-Я]', '', word)


def preprocess_text(text):
    text = re.sub(r'[0-9]+[,.]?[0-9]*\s?(ккал|г|грамм|мг|калорий|литров|мл|%|/100)', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = ' '.join(clean_non_alphanumeric(word) for word in text.split())
    return text


def extract_morphemes(text):
    words = text.split()
    return {cached_parse(word) for word in words if len(word) > 2}


def load_database():
    with open('../db/names.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
    preprocessed_lines = [preprocess_text(line) for line in lines]

    with open('../db/data.txt', 'r', encoding="utf-8") as f:
        data_lines = f.readlines()

    return data_lines, preprocessed_lines


def create_inverted_index(preprocessed_lines):
    inverted_index = {}
    for idx, line in enumerate(preprocessed_lines):
        words = extract_morphemes(line)
        for word in words:
            if word in inverted_index:
                inverted_index[word].add(idx)
            else:
                inverted_index[word] = {idx}
    return inverted_index


def search_lines(user_query, original_lines, inverted_index, max_per_word=10):
    """
    Поиск строк, содержащих слова из пользовательского запроса.
    Для каждого слова выбирается до max_per_word случайных строк.
    """
    preprocessed_query = preprocess_text(user_query)
    query_words = extract_morphemes(preprocessed_query)

    stop_words = {'приготовить'}

    additional_info = []

    for word in query_words:
        if word in stop_words:
            continue
        if word in inverted_index:
            matching_indices = list(inverted_index[word])
            if len(matching_indices) > max_per_word:
                sampled_indices = random.sample(matching_indices, max_per_word)
            else:
                sampled_indices = matching_indices
            for idx in sampled_indices:
                line = original_lines[idx].strip()
                additional_info.append(line)

    return additional_info


def main():
    original_lines, preprocessed_lines = load_database()

    inverted_index = create_inverted_index(preprocessed_lines)

    user_query = input("Введите ваш запрос: ")

    matched_lines = search_lines(user_query, original_lines, inverted_index, 5)

    additional_info = "\n".join(matched_lines)

    prompt = f"Ваш запрос: {user_query}\nДополнительная информация:\n{additional_info}\nОтвет:"

    print(prompt)
    print()

if __name__ == "__main__":
    main()
