from sentence_transformers import SentenceTransformer
def get_embedding_function():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model

embedding_function = get_embedding_function()
query_embedding = embedding_function.encode('Бургер')
document_embedding = embedding_function.encode("Кресс-салат")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print(query_embedding, document_embedding)
print(cosine_similarity(np.array(query_embedding).reshape(1,-1), np.array(document_embedding).reshape(1,-1)))

#similarity = cosine_similarity([query_embedding], [document_embedding])
#print(f"Косинусное сходство между 'Бургер' и 'Шницель': {similarity[0][0]}")
