from importlib.resources import path
import itertools
from contextlib import redirect_stderr
from distutils.text_file import TextFile
from collections import defaultdict
from operator import truediv
from webbrowser import get
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import nltk
from os import walk
import os
from flask import Flask, render_template, request,Markup
from flask_wtf import FlaskForm
#from flaskext.markdown import Markdown
from wtforms import SubmitField, TextAreaField
import spacy
from spacy import displacy
import torch
from transformers.file_utils import is_tf_available,is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast,BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/upload", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        files = request.files['filename']
        files_pash = "upload/"+files.filename
        files.save(files_pash)
        search_word=request.form.get('wordsearch')
    
    f=count()
    articles=[]
    for i in range(len(f)):
        x = open(f"upload/"+f[i], "r",encoding='cp932', errors='ignore')
        article = x.read()
        tokens = [w for w in word_tokenize(article.lower()) if w.isalpha()]
        no_stops = [t for t in tokens if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)
    dictionary = Dictionary(articles)
    ws = dictionary.token2id.get(search_word)
    corpus = [dictionary.doc2bow(a) for a in articles]
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count
    swc = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)
    doc = corpus[0]
    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    stw = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

    return render_template("ans.html",a=files_pash,tfidf1=stw[0],tfidf2=stw[1],tfidf3=stw[2],tfidf4=stw[3],tfidf5=stw[4],wc1=swc[0],wc2=swc[1],wc3=swc[2],wc4=swc[3],wc5=swc[4],searchword=ws,word=search_word)
def count():
    f=[]
    path = "upload/"
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
    return(f)
@app.route("/show", methods=['GET', 'POST'])
def show_text():
    if request.method == "POST" and request.form['show'] == 'show':
        f=count()
        if f==[]:
            return render_template("index.html",articles="ไม่มีไฟล์ให้แสดง!")
        
        search_word = request.form.getlist("op")

        all_t=all_text_nlp(f)
        options2 = {"ents": search_word}
        text_nlp=displacy.render(all_t, style="ent",options=options2)
        articles=[]
        for i in range(len(f)):
            x = open(f"upload/"+f[i], "r",encoding='cp932', errors='ignore')
            article = x.read()
            tokens = [w for w in word_tokenize(article.lower()) if w.isalpha()]
            no_stops = [t for t in tokens if t not in stopwords.words('english')]
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
            articles.append(lemmatized)
        dictionary = Dictionary(articles)
        corpus = [dictionary.doc2bow(a) for a in articles]
        total_word_count = defaultdict(int)
        for word_id, word_count in itertools.chain.from_iterable(corpus):
            total_word_count[word_id] += word_count
        swc = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)
        doc = corpus[0]
        tfidf = TfidfModel(corpus)
        tfidf_weights = tfidf[doc]
        stw = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
        TF_IDF=[]
        for term_id, weight in stw[:5]:
                x=dictionary.get(term_id), weight
                TF_IDF.append(x)
        bow=[]
        for word_id, word_count in swc[:5]:
            x=dictionary.get(word_id), word_count
            bow.append(x)
            
        return render_template("index.html",articles=Markup(text_nlp),bow=bow,TF_IDF=TF_IDF)
@app.route("/search", methods=['GET', 'POST'])
def search_words():
    search_word = request.form.get("wordsearch")
    
    f=count()
    if f==[]:
        return render_template("index.html",articles="ไม่มีไฟล์ให้แสดง!")
    
    articles=[]
    for i in range(len(f)):
        x = open(f"upload/"+f[i], "r",encoding='cp932', errors='ignore')
        article = x.read()
        tokens = [w for w in word_tokenize(article.lower()) if w.isalpha()]
        no_stops = [t for t in tokens if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)
    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count
    swc = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)
    doc = corpus[0]
    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    stw = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
    TF_IDF=[]
    for term_id, weight in stw[:5]:
            x=dictionary.get(term_id), weight
            TF_IDF.append(x)
    bow=[]
    for word_id, word_count in swc[:5]:
        x=dictionary.get(word_id), word_count
        bow.append(x)

    word_search=search_words(dictionary,articles,stw,search_word)
    doc = nlp(search_word)
    text_nlp=displacy.render(doc, style="ent")

    return render_template("index.html",articles=Markup(text_nlp),bow=bow,TF_IDF=TF_IDF,search_word=word_search) 
@app.route("/Model",methods = ['GET','POST'])
def fakenews():
    if request.method == 'POST':
        faketext = request.form['text_area']
        realnews = get_prediction(faketext, convert_to_label=True)
        return render_template("index.html",model_T = realnews) 
def get_prediction(text, convert_to_label=False):
    model_path = "fake-news-bert-base-uncased"
    model=AutoModelForSequenceClassification.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512,return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())
    
def all_text_nlp(f):
    articles=[]
    for i in range(len(f)):
        x = open(f"upload/"+f[i], "r",encoding='cp932', errors='ignore')
        article = x.read()
        doc = nlp(article)
        articles.append(doc)
  
    return articles
def search_words(dictionary,articles,stw,search_word):
    ID_words = dictionary.token2id.get(search_word)
    
    if ID_words ==None :
        Ans="ไม่เจอคำที่ค้นหา"
        return (Ans)
    else:
        ID_words = dictionary.token2id.get(search_word)
        
        corpus = [dictionary.doc2bow(a) for a in articles]
        
        total_word_count = defaultdict(int)
        for word_id, word_count in itertools.chain.from_iterable(corpus):
            total_word_count[word_id] += word_count
        
        doc = corpus[0]
        tfidf = TfidfModel(corpus)
        tfidf_weights = tfidf[doc]   
        
        t1="search text : "+dictionary.get(ID_words)
        t2="\nID Words :"+str(ID_words)
        t3="\nBOW :"+str(total_word_count[ID_words])
        try:
            tfidf_weights[ID_words]
        except:
            t4="\nTF-IDF : ค่า TF-IDF หาไม่ได้"
        else:
            t4="\nTF-IDF :"+str(stw[total_word_count[ID_words]])
        Ans= t1+t2+t3+t4
    return(Ans)
if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)