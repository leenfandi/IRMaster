import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.sparse as sp
import joblib 
from nltk.tokenize import word_tokenize

def vectorize_documents(documents):
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', norm='l2')
    doc_vectors = vectorizer.fit_transform(documents)
    return vectorizer, doc_vectors

def save_document_vectors(vectorizer, doc_vectors, file_path):
    joblib.dump(doc_vectors, file_path)
    print(f"Document vectors saved to '{file_path}' with shape: {doc_vectors.shape}")

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    return file_content

def split_entries(file_content, delimiter):
    entries = file_content.split(delimiter)
    return entries

def final_dict(entries):
    final_dict = {}
    for index, entry in enumerate(entries):
        lis = str(index + 1)  # Using index + 1 as a simple identifier
        string = ' '.join(word_tokenize(entry))
        final_dict[lis] = string
    return final_dict

file_path = 'StopwordProccess.npy'
sample_content = read_file_content(file_path)[:500]  
print(sample_content) 


file_content = read_file_content(file_path)

delimiter = '\n' 
entries = split_entries(file_content, delimiter)

final_dict = final_dict(entries)
all_documents = list(final_dict.values())

# Verify the number of documents
print(f"Number of documents: {len(all_documents) }") 

vectorizer, doc_vectors = vectorize_documents(all_documents)

joblib_file = "doc_vector_sparse.pkl"
save_document_vectors(vectorizer, doc_vectors, joblib_file)

print(f"Document vectors saved to '{joblib_file}' with shape: {doc_vectors.shape}")

# will print like this [
# Number of documents: 403457
# Document vectors saved to 'doc_vector_sparse.pkl' with shape: (403457, 589213) ]