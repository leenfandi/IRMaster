from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from init import app
import init
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
import json
from query_processing import process_query
from flask import Flask
# app = Flask(__name__)

# def matching(doc_vector, query_vector):
#     # enc = OneHotEncoder(handle_unknown='ignore')
#     # array_vec_1 = np.array(doc_vector, dtype=object)
#     # array_vec_2 = np.array(query_vector, dtype=object)
# Function to preprocess the query

def process_query(query):
    processed_query = query.lower() 
    return processed_query

def matching(doc_vector, query_vector):
    cosine_similarities = cosine_similarity(doc_vector, query_vector).flatten()
    related_doc_id = cosine_similarities.argsort()[:-10:-1]
    return related_doc_id


tfidf_vectorizer = TfidfVectorizer()
doc_vector = tfidf_vectorizer.fit_transform()

joblib.dump(tfidf_vectorizer, 'C:/Users/Asus/IRMaster/doc_vector_sparse.pkl')

queries = [
    "u get rid corn toe",
    "how to clean antique furniture",
    "tips for restoring old paintings"
]

preprocessed_queries = [process_query(query) for query in queries]
query_vectors = tfidf_vectorizer.transform(preprocessed_queries)

query_vectors = csr_matrix(query_vectors)
tfidf_vectorizer = joblib.load('C:/Users/Asus/IRMaster/doc_vector_sparse.pkl')

query_vectors = tfidf_vectorizer.transform(preprocessed_queries)
query_vectors = csr_matrix(query_vectors)

queries_id = {}
for i, query_vector in enumerate(query_vectors):
    related_doc_id = matching(doc_vector, query_vector)
    queries_id[i] = related_doc_id.tolist()

output_file = 'C:/Users/Asus/IRMaster/queries_id.json'
with open(output_file, 'w') as f:
    json.dump(queries_id, f, indent=4)

print(f"queries_id dictionary saved to {output_file}.")


