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
# Expend corpus

# app = Flask(__name__)


def expend_corpus(corpusName):
    fin = open(corpusName+".tsv", "rt", encoding="utf8")
    fout = open("expended_"+corpusName+".tsv", "wt", encoding="utf8")
    for line in fin:
        fout.write(line.replace(line, contractions.fix(line)))
    fin.close()
    fout.close()


def Tokenization(corpusName):
    dic = {}
    with open(corpusName+".tsv", encoding="utf8")as file:
        tsv_file = csv.reader(file, delimiter="\t")
        i = 0
        for line in tsv_file:
            i += 1
            word_tokens = word_tokenize(line[1].strip())
            dic[line[0]] = word_tokens
    return dic


# date form


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def replace_all(text, mydict):
    for gb, us in mydict.items():
        text = text.replace(us, gb)
    return text


def create_acronym_dic(acronymFileName):
    acronym = {}
    with open(acronymFileName+".csv", encoding="utf8")as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            acronym[line[0]] = line[1]
    return acronym


def replace_acronym(text, mydict):
    x = ''
    for key in mydict:
        if text == key:
            return mydict[key]
        else:
            x = text
    return x


def Clean_tokens(dic):
    read_dictionary = np.load('us2gb.npy', allow_pickle='TRUE').item()
    Clean_tokens = {}
    for lis in dic:
        #             print(lis)
        #             print("-----")
        filtered_sentence = []
        for w in dic[lis]:
            try:
                if w.isnumeric():
                    filtered_sentence.append(num2words.num2words(w))
                elif is_date(w):
                    d = parser.parse(w)
                    filtered_sentence.append(d.strftime("%Y-%m-%d"))
                else:
                    test_str = w.translate(
                        str.maketrans('', '', string.punctuation))
                    if test_str != "":
                        uni = anyascii(test_str)
                        res = replace_all(uni, read_dictionary)
        #                 acron=replace_acronym(res.upper(),acronym)
                        filtered_sentence.append(res.lower())
            except:
                filtered_sentence.append(w)
                pass

        Clean_tokens[lis] = filtered_sentence
    return Clean_tokens
# Type Lematization


def POSdic(dic):
    posDic = {}
    for key in dic:
        tagged = nltk.pos_tag(dic[key])
        posDic[key] = tagged
    return posDic


def penn2morphy(penntag, returnNone=True):
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def lemetization_dic(posDic):
    lemmaDic = {}
    lemmatizer = WordNetLemmatizer()
    for key in posDic:
        lemma_sentence = []
        for a, b in posDic[key]:
            posValue = penn2morphy(b)
            if posValue == None:
                value = lemmatizer.lemmatize(a)
                lemma_sentence.append(value)
            else:
                value = lemmatizer.lemmatize(a, pos=posValue)
                lemma_sentence.append(value)
            lemmaDic[key] = lemma_sentence
    return lemmaDic


def stopwordsDic(lemmaDic):
    stop_words = set(stopwords.words('english'))
    stopWordDic = {}
    for lis in lemmaDic:
        filtered_sentence = []
        for w in lemmaDic[lis]:
            if w not in stop_words:
                filtered_sentence.append(w)
        stopWordDic[lis] = filtered_sentence
    # one line statement for the above operation
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    # this take words like not ,no as a stop words!!!
    return stopWordDic


def acronymDic(acronymDic):
    states_dictionary = np.load('acronyms.npy', allow_pickle='TRUE').item()
    countryDic = {}
    for lis in acronymDic:
        filtered_sentence = []
        for w in acronymDic[lis]:
            word = replace_acronym(w, states_dictionary)
            filtered_sentence.append(word.lower())
        countryDic[lis] = filtered_sentence
    return countryDic


def CountryDic(acronymDic):
    country_dictionary = np.load('country.npy', allow_pickle='TRUE').item()
    countryDic = {}
    for lis in acronymDic:
        filtered_sentence = []
        for w in acronymDic[lis]:
            word = replace_acronym(w, country_dictionary)
            filtered_sentence.append(word.lower())
        countryDic[lis] = filtered_sentence
    return countryDic


def statesDic(acronymDic):
    states_dictionary = np.load('states.npy', allow_pickle='TRUE').item()
    countryDic = {}
    for lis in acronymDic:
        filtered_sentence = []
        for w in acronymDic[lis]:
            word = replace_acronym(w, states_dictionary)
            filtered_sentence.append(word.lower())
        countryDic[lis] = filtered_sentence
    return countryDic


def finalDic(StatesDic):
    Fdic = {}
    for lis in StatesDic:
        string = ''
        for w in StatesDic[lis]:
            string = string+' '+w
        Fdic[str(lis)] = string
    return Fdic


# def normalization():
    # # expended_dic = expend_corpus(corpus)
    # token_dic = Tokenization(corpus)
    # # print("######################################")
    # stop_dic = stopwordsDic(token_dic)
    # # print("######################################")
    # acron_dic = acronymDic(stop_dic)
    # # print("######################################")
    # # np.save("AcronymProccess.npy",acron_dic)
    # countryDic = CountryDic(acron_dic)
    # # print("######################################")
    # # np.save("CountryProccess.npy",countryDic)
    # # print("######################################")
    # StatesDic = statesDic(countryDic)
    # # print("######################################")
    # clean_dic = Clean_tokens(StatesDic)
    # # print("######################################")
    # POS_dic = POSdic(clean_dic)
    # # print("######################################")
    # lema_dic = lemetization_dic(POS_dic)
    # # print("######################################")
    # # print("######################################")
    # stop_dic = stopwordsDic(lema_dic)
    # Final = finalDic(stop_dic)
    # return Final

    # return normalizeResualt
# print(stop_dic)
