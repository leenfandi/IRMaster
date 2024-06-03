import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import representation
import query_processing as qp
import csv


def recall_at_k(relevant_set, retrieved_set):
    """
    Calculates the recall at k for the retrieved set given the relevant set.
    """
    k = len(retrieved_set)
    relevant_docs = set(relevant_set)
    retrieved_docs = set(retrieved_set)
    relevant_and_retrieved_docs = relevant_docs.intersection(retrieved_docs)
    recall = len(relevant_and_retrieved_docs) / len(relevant_docs)
    return recall
#relevant_documents=[1,2,3,4,5,6,7,8]
#result_docs=[1,3,4,9,10]
#tp=[1,3,4]
#recall= len(tp)/len(relevant_documents)
#recall= 3/8


def precision_at_k(relevant_set, retrieved_set):
    """
    Calculates the precision at k for the retrieved set given the relevant set.
    """
    k = len(retrieved_set)
    relevant_docs = set(relevant_set)
    retrieved_docs = set(retrieved_set)
    relevant_and_retrieved_docs = relevant_docs.intersection(retrieved_docs)
    precision = len(relevant_and_retrieved_docs) / k
    return precision
#relevant_documents=[1,2,3,4,5,6,7,8]
#retriev_docs=[1,3,4,9,10]
#tp=[1,3,4]
#precision= len(tp)/len(retriev_docs)
#precision= 3/5

def precison_recall_query(relevant_docs, retrieved_docs):
    precision = precision_at_k(relevant_docs, retrieved_docs)
    recall = recall_at_k(relevant_docs, retrieved_docs)
    return precision, recall


def relevent_docs():
    dic = {}
    with open("qrelsW", "r") as file:
        tsv_file = file.readlines()
        for line in tsv_file:
            lista = line.split()
            dic.setdefault(lista[0], [])
            dic[lista[0]].append(lista[2])
#                 print( dic[lista[0]])
    return dic


rel = relevent_docs()


def relevent_queries():
    dic = {}
    with open("queriesW.csv") as file:
        tsv_file = csv.reader(file, delimiter=",")
        for line in tsv_file:

            #                 print (line[0])
            dic[line[0]] = line[1]
#                 print( dic[lista[0]])
    return dic


queries = relevent_queries()


def finalDic(StatesDic):
    Fdic = {}
    for lis in StatesDic:
        string = ''
        for w in StatesDic[lis]:
            string = string+' '+w
        Fdic[str(lis)] = string
    return Fdic


stemm = np.load('DOC_SwordsProccess.npy', allow_pickle='TRUE').item()
Final = finalDic(stemm)


documents = list(Final.values())
vectorizer = TfidfVectorizer(
    sublinear_tf=True, stop_words='english', norm='l2', max_df=0.9)

# Fit the vectorizer to the documents
# tfidf_matrix = vectorizer.fit_transform(documents)
vectorizer.fit(documents)
doc_vector = vectorizer.transform(documents)


def retrieve_documents(query):
    qe = qp.query_process(query)
    query_vector = vectorizer.transform([qe])
    cosinesimilarities = cosine_similarity(doc_vector, query_vector).flatten()
    related_doc_id = cosinesimilarities.argsort()[:-11:-1]
    # retrieved documents for query q
    ides = []
    for i in related_doc_id:
        x = list(Final)[i]
        ides.append(x)
    return ides


def retrieve_documents1(query):
    qe = qp.query_process(query[0])
    query_vector = vectorizer.transform([qe])
    cosinesimilarities = cosine_similarity(doc_vector, query_vector).flatten()
    related_doc_id = cosinesimilarities.argsort()[:-11:-1]
    # retrieved documents for query q
    ides = []
    for i in related_doc_id:
        x = list(Final)[i]
        ides.append(x)
    return ides


def evaluation(queries):
    dic = {}
    with open("P&R@10.txt", 'wt') as fr:
        for key in queries:
            relevent = rel[key]
            query = queries[key]
            retrieved = retrieve_documents(query)
            lis = precison_recall_query(relevent, retrieved)
            print(lis)
            fr.write(str(key) + ":" + str(lis)+"\n")
            dic[key] = lis
    return dic


eva = evaluation(queries)


def binary_vector(relevent, retrieved):
    resualt = []
    for i in retrieved:
        if i in relevent:
            resualt.append(1)
        else:
            resualt.append(0)
    return resualt


def ground_truths():
    dic = []
    for key in queries:
        relevent = rel[key]
#         print(relevent)
        query = queries[key]
        retrieved = retrieve_documents(query)
#         print(retrieved)
        bV = binary_vector(relevent, retrieved)
#         print(bV)
        dic.append(bV)
#         print("------")
    return dic


gt = ground_truths()


def retrieval_results():
    dic = []
    for key in queries:
        relevent = rel[key]
        query = queries[key]
        retrieved = retrieve_documents(query)
#         print(retrieved)
        dic.append(retrieved)
    return dic


rr = retrieval_results()


def calculate_ap(ground_truth, retrieval_results):
    precision = []
    relevant_count = 0
    for i, doc in enumerate(retrieval_results):
        x = retrieval_results.index(doc)
        if ground_truth[x] == 1:
            relevant_count += 1
            precision.append(relevant_count / (i+1))
    try:
        average_precision = sum(precision) / len(precision)
    except:
        average_precision = 0.1
        pass
    return average_precision


print(calculate_ap(gt[1], rr[1]))


def calculate_map(ground_truths, retrieval_results):
    total_precision = 0
    for i, ground_truth in enumerate(ground_truths):
        ap = calculate_ap(ground_truth, retrieval_results[i])
        total_precision += ap
    map = total_precision / len(ground_truths)
    return map


fMAP = calculate_map(gt, rr)
print(fMAP)


def reciprocal_rank(ground_truth, retrieval_results):
    reciprocal_rank = 0
    for i, doc in enumerate(retrieval_results):
        x = retrieval_results.index(doc)
        if ground_truth[x] == 1:
            reciprocal_rank = 1 / (i+1)
            break
    return reciprocal_rank


def calculate_mrr(ground_truths, retrieval_results):
    reciprocal_ranks = []
    for i, ground_truth in enumerate(ground_truths):
        rr = reciprocal_rank(ground_truth, retrieval_results[i])
        reciprocal_ranks.append(rr)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr


fMrr = calculate_mrr(gt, rr)
print(fMrr)
# Example usage

#ranks = [1, 0, 3, 0, 2]
#reciprocal_ranks = [1 / 1, 1 / 3, 1 / 2] = [1.0, 0.3333333333333333, 0.5]
#mrr 0.611111111111111

#ranks = [3, 1, 1, 2, 1]
#reciprocal_ranks = [1 / 3, 1 / 1, 1 / 2, 1 / 1] = [0.3333333333333333, 1.0, 0.5, 1.0]
#mrr = (0.3333333333333333 + 1.0 + 0.5 + 1.0) / 4 = 0.7083333333333334