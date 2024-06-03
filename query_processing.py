from flask import Flask
import math
import nltk
import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re
import num2words
import contractions
from textblob import Word
import string
from dateutil.parser import parse
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import csv
from anyascii import anyascii
from breame.spelling import get_american_spelling, get_british_spelling
from dateutil.parser import parse
import numpy as np
from argparse import ArgumentParser
from dateutil import parser
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# app = Flask(__name__)


def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def replace_all(text, mydict):
    for gb, us in mydict.items():
        text = text.replace(us, gb)
    return text


def penn2morphy(penntag, returnNone=True):
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def replace_acronym(text, mydict):
    x = ''
    for key in mydict:
        if text == key:
            return mydict[key]
        else:
            x = text
    return x


def Clean_queriess(lis):
    read_dictionary = np.load('us2gb.npy', allow_pickle='TRUE').item()
    filtered_sentence = []
    for w in lis:
        if w.isnumeric():
            filtered_sentence.append(num2words.num2words(w))
        elif is_date(w):
            d = parser.parse(w)
            filtered_sentence.append(d.strftime("%Y-%m-%d"))
        else:
            test_str = w.translate(str.maketrans('', '', string.punctuation))
            if test_str != "":
                uni = anyascii(test_str)
                res = replace_all(uni, read_dictionary)
        #                 acron=replace_acronym(res.upper(),acronym)
                filtered_sentence.append(res)
    return filtered_sentence


def POSList(lis):
    listt = []
#     for key in lis:
    tagged = nltk.pos_tag(lis)
    return tagged


def lemetization_list(posDic):
    lemmatizer = WordNetLemmatizer()
    lemma_sentence = []
    for a, b in posDic:
        posValue = penn2morphy(b)
        if posValue == None:
            value = lemmatizer.lemmatize(a)
            lemma_sentence.append(value)
        else:
            value = lemmatizer.lemmatize(a, pos=posValue)
            lemma_sentence.append(value)
    return lemma_sentence


def stopwordslis(lemmaDic):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in lemmaDic:
        if w not in stop_words:
            filtered_sentence.append(w)
        # one line statement for the above operation
        # filtered_sentence = [w for w in word_tokens if not w in stop_words]
        # this take words like not ,no as a stop words!!!
    return filtered_sentence


def acronymlist(lis):
    states_dictionary = np.load('acronyms.npy', allow_pickle='TRUE').item()
    filtered_sentence = []
    for w in lis:
        word = replace_acronym(w, states_dictionary)
        filtered_sentence.append(word.lower())
    return filtered_sentence


def Countrylis(acronymDic):
    country_dictionary = np.load('country.npy', allow_pickle='TRUE').item()
    filtered_sentence = []
    for w in acronymDic:
        word = replace_acronym(w, country_dictionary)
        filtered_sentence.append(word.lower())
    return filtered_sentence


def stateslis(acronymDic):
    states_dictionary = np.load('states.npy', allow_pickle='TRUE').item()
    filtered_sentence = []
    for w in acronymDic:
        word = replace_acronym(w, states_dictionary)
        filtered_sentence.append(word.lower())
    return filtered_sentence


def query_process(q):
    expanded_words = []
    for word in q.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    word_tokens = word_tokenize(expanded_text.strip())
    stop_dic = stopwordslis(word_tokens)
    acron_dic = acronymlist(stop_dic)
    countryDic = Countrylis(acron_dic)
    StatesDic = stateslis(countryDic)
    clean_list = Clean_queriess(StatesDic)
    POS_dic = POSList(clean_list)
    lema_dic = lemetization_list(POS_dic)
    stop_dic = stopwordslis(lema_dic)
    res = ' '.join(stop_dic)
    # vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    # query_vector = vectorizer.transform([res])
    return res


# Create a list of documents
# def vectorize_query(processed_query):
#     vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
#     query_vector = vectorizer.transform([processed_query])
#     return query_vector
