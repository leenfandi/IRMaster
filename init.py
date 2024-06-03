from normalization import *
from representation import *
from query_processing import *
from matching import *
from flask import Flask
from flask import request, redirect, jsonify
from flask import render_template
from flask import url_for, json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from logging import FileHandler, WARNING
import speech_recognition as sr
from googletrans import Translator
app = Flask(__name__, template_folder='templates')


# dataset1after = np.load('StopwordProccess.npy', allow_pickle='TRUE').item()
# dataset2after = np.load('DOC_SwordsProccess.npy',allow_pickle='TRUE').item()
#
# dataset1original = np.load('corpusDic.npy', allow_pickle='TRUE').item()
# dataset2original = np.load('DOC_corpusDic.npy', allow_pickle='TRUE').item()

DATAAFTERNORMALIZATION = np.load(
    'StopwordProccess.npy', allow_pickle='TRUE').item()
ORIGINALDATA = np.load('corpusDic.npy', allow_pickle='TRUE').item()
FINAL = finalDic(DATAAFTERNORMALIZATION)
DOCUMENT = list(FINAL.values())
DOCVECTOR = vectorize_dic(DOCUMENT)
LANGUAGE = "english"


@app.route('/', methods=["GET"])
def index():

    return render_template('index.html')


@app.route('/processquery', methods=["GET", "POST"])
def processquery():
    try:
        query = request.json['data']
        if (LANGUAGE == "arabic"):
            translator = Translator()
            query = translator.translate(str(query), dest='en').text
        processed_query = query_process(str(query))
        query_vector = vectorize_query(processed_query)
        ides = matching(DOCVECTOR, query_vector)
        related_doc_id = []
        for i in ides:
            x = list(FINAL)[i]
            related_doc_id.append(x)
        result = []
        for i in related_doc_id:
            data = ORIGINALDATA[i]
            result.append(data)
        print(result)
        if (LANGUAGE == "arabic"):
            translated = []
            for i in result:
                p = translator.translate(str(i), dest='ar').text
                translated.append(p)
            return jsonify(translated)
        return jsonify(result)
    except:
        return jsonify(" ")


@app.route('/writeprocessquery', methods=["POST"])
def writeprocessquery():
    if request.method == "POST":
        query = request.form.get("fname")
        try:
            if query != "":
                if (LANGUAGE == "arabic"):
                    translator = Translator()
                    query = translator.translate(str(query), dest='en').text
                    # print(query)
                processed_query = query_process(str(query))
                query_vector = vectorize_query(processed_query)
                ides = matching(DOCVECTOR, query_vector)
                related_doc_id = []
                for i in ides:
                    x = list(FINAL)[i]
                    related_doc_id.append(x)
                result = []
                for i in related_doc_id:
                    # data = documents[i]
                    data = ORIGINALDATA[i]
                    result.append(data)
                print(result)
                if (LANGUAGE == "arabic"):
                    translated = []
                    for i in result:
                        p = translator.translate(str(i), dest='ar').text
                        translated.append(str(p))
                        # print({'name': translated})
                    return jsonify({'name': translated})
                return jsonify({'name': result})
            else:
                return jsonify({'name': ""})
        except:
            return jsonify({'name': " "})


@app.route('/my_page', methods=['POST', 'GET'])
def my_page(result):
    print(result)
    return jsonify(result)


# corpusFinal = np.load('corpusDic.npy', allow_pickle='TRUE').item()
# normalizeResualt = np.load('StopwordProccess.npy', allow_pickle='TRUE').item()


@app.route('/submit-option', methods=['POST'])
def submit_option():
    selected_option = request.args.get('selected_option', '')
    if selected_option == "Dataset1":
        global DATAAFTERNORMALIZATION
        DATAAFTERNORMALIZATION = np.load(
            'StopwordProccess.npy', allow_pickle='TRUE').item()
        global ORIGINALDATA
        ORIGINALDATA = np.load('corpusDic.npy', allow_pickle='TRUE').item()
        global FINAL
        FINAL = finalDic(DATAAFTERNORMALIZATION)
        global DOCUMENT
        DOCUMENT = list(FINAL.values())
        global DOCVECTOR
        DOCVECTOR = vectorize_dic(DOCUMENT)
    else:

        DATAAFTERNORMALIZATION = np.load(
            'DOC_SwordsProccess.npy', allow_pickle='TRUE').item()
        ORIGINALDATA = np.load('DOC_corpusDic.npy', allow_pickle='TRUE').item()
        FINAL = finalDic(DATAAFTERNORMALIZATION)
        DOCUMENT = list(FINAL.values())
        DOCVECTOR = vectorize_dic(DOCUMENT)
    return selected_option
    # return 'Selected option: {}'.format(selected_option)


@app.route('/submit-lang-option', methods=['POST'])
def submit_lang_option():
    selected_option = request.args.get('selected_option', '')
    if selected_option == "en-US":
        global LANGUAGE
        LANGUAGE = "english"
    else:
        LANGUAGE = "arabic"
    print(selected_option)
    return selected_option


if __name__ == '__main__':
    app.debug = True
    app.run(port=5000)
